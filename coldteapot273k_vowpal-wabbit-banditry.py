from vowpalwabbit import pyvw
import random
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas as pd
import itertools
# VW tries to minimize loss/cost, therefore we will pass cost as -reward

CUSTOMER_PURCHASED_PRODUCT = -1.0
CUSTOMER_DID_NOT_PURCHASE_PRODUCT = 0.0
def get_cost(context,action):
    if context['user'] == "Client_type_1":
        if context['quarter'] == "2nd" and action == 'credit_large':
            return CUSTOMER_PURCHASED_PRODUCT
        elif context['quarter'] == "4th" and action == 'deposit_medium':
            return CUSTOMER_PURCHASED_PRODUCT
        else:
            return CUSTOMER_DID_NOT_PURCHASE_PRODUCT
    elif context['user'] == "Client_type_2":
        if context['quarter'] == "2nd" and action == 'deposit_small':
            return CUSTOMER_PURCHASED_PRODUCT
        elif context['quarter'] == "4th" and action == 'credit_large':
            return CUSTOMER_PURCHASED_PRODUCT
        else:
            return CUSTOMER_DID_NOT_PURCHASE_PRODUCT
# This function modifies (context, action, cost, probability) to VW friendly format
def to_vw_example_format(context, actions, cb_label = None):
    if cb_label is not None:
        chosen_action, cost, prob = cb_label
    example_string = ""
    example_string += "shared |User user={} quarter={}\n".format(context["user"], context["quarter"])
    for action in actions:
        if cb_label is not None and action == chosen_action:
            example_string += "0:{}:{} ".format(cost, prob)
        example_string += "|Action product={} \n".format(action)
    #Strip the last newline
    return example_string[:-1]
context = {"user":"Client_type_1","quarter":"2nd"}
actions = ["credit_large", "deposit_small", "deposit_medium", "leasing_medium"]

print(to_vw_example_format(context,actions))
def sample_custom_pmf(pmf):
    total = sum(pmf)
    scale = 1/total
    pmf = [x * scale for x in pmf]
    draw = random.random()
    
    print(f'random draw threshold - {draw}')
    
    sum_prob = 0.0
    for index, prob in enumerate(pmf):
        
        print(f'sum_prob - {sum_prob}')
        print(f'index - {index}')
        print(f'prob - {prob}')
        
        sum_prob += prob
        if(sum_prob > draw):
            
            print(f'sum_prob[={sum_prob}] > draw[={draw}], sample action={index} with prob={prob} \n ==============')
            
            return index, prob
def get_action(vw, context, actions):
    vw_text_example = to_vw_example_format(context,actions)
    pmf = vw.predict(vw_text_example)
    
#     print(f'pmf(predict raw output) - {pmf}')
    pmf_record = pmf
    
    chosen_action_index, prob = sample_custom_pmf(pmf)
    return actions[chosen_action_index], prob, pmf_record
users = ['Client_type_1', 'Client_type_2']
times = ['2nd', '4th']
actions = ["credit_large", "deposit_small", "deposit_medium", "leasing_medium", "deposit_large", "credit_medium", "credit_small"]

def choose_user(users):
    return random.choice(users)

def choose_time(times):
    return random.choice(times)

# display preference matrix
def get_preference_matrix(cost_fun):
    def expand_grid(data_dict):
        rows = itertools.product(*data_dict.values())
        return pd.DataFrame.from_records(rows, columns=data_dict.keys())

    df = expand_grid({'users':users, 'times': times, 'actions': actions})
    df['cost'] = df.apply(lambda r: cost_fun({'user': r[0], 'quarter': r[1]}, r[2]), axis=1)

    return df.pivot_table(index=['users', 'times'], 
            columns='actions', 
            values='cost')

get_preference_matrix(get_cost)
abs(get_preference_matrix(get_cost))
ctxt = {'users':'Client_type_1', 'times': '2nd'}

# list(get_preference_matrix(get_cost).loc['Client_type_1', '2nd', :].values.ravel()) #.tolist()
list(get_preference_matrix(get_cost).loc[tuple(ctxt.values()), :].values.ravel())
tuple(ctxt.values())
def run_simulation(vw, num_iterations, users, times, actions, cost_function, do_learn = True):
    cost_sum = 0.
    scr = []
    
    records = {
            'iteration': list(),
#             'context': list(),
            'action': list(),
            'prob': list(),
            'cost': list(),
            'cost_sum': list(),
            'scr': list(),
            'predicted_preference': list(), # = 'pmf'
            'true_preference': list()
        }

    for i in range(1, num_iterations+1):
        # 1. In each simulation choose a user
        user = choose_user(users)
        # 2. Choose time of day for a given user
        time = choose_time(times)

        # 3. Pass context to vw to get an action
        context = {'user': user, 'quarter': time}
        action, prob, pmf_record = get_action(vw, context, actions)
        
#         print(f'user - {user} \n time - {time}')
#         print(f'action - {action} \n prob - {prob}')

        # 4. Get cost of the action we chose
        cost = cost_function(context, action)
        cost_sum += cost
        
#         print(f'cost - {cost} \n cost_sum - {cost_sum} \n ==============')

        if do_learn:
            # 5. Inform VW of what happened so we can learn from it
            vw_format = vw.parse(to_vw_example_format(context, actions, (action, cost, prob)),pyvw.vw.lContextualBandit)
            # 6. Learn
            vw.learn(vw_format)

        # We negate this so that on the plot instead of minimizing cost, we are maximizing reward
        scr.append(-1*cost_sum/i)
        
        
        
        # logging:
        
        records['iteration'].append(i),
        
        for key in context.keys():
    
            records.setdefault(key, [])
            records[key].append(context[key])
            
#             'context': context,
        records['action'].append(action),
        records['prob'].append(prob),
        records['cost'].append(cost),
        records['cost_sum'].append(cost_sum)
        records['scr'] = scr
        records['predicted_preference'].append([round(x, 1) for x in pmf_record]) # estimated preference
        records['true_preference'].append(list(abs(get_preference_matrix(cost_function)).loc[tuple(context.values()), :].values.ravel()))

    return scr, records
def plot_scr(num_iterations, scr):
    plt.plot(range(1,num_iterations+1), scr)
    plt.xlabel('num_iterations', fontsize=14)
    plt.ylabel('scr', fontsize=14)
    plt.ylim([0,1])
# Instantiate learner in VW
# vw = pyvw.vw("--cb_explore_adf -q UA --quiet --epsilon 0.2")
# vw = pyvw.vw("--cb_explore_adf -q UA --epsilon 0.005")
vw = pyvw.vw("--cb_explore_adf -q UA --epsilon 0.00 --audit")

num_iterations = 5000
# num_iterations = 50
scr, records = run_simulation(vw, num_iterations, users, times, actions, get_cost)

plot_scr(num_iterations, scr)
pd.DataFrame.from_dict(records, orient='columns')#.head(15)
# Instantiate learner in VW but without -q
vw = pyvw.vw("--cb_explore_adf --quiet --epsilon 0.2")

num_iterations = 5000
scr = run_simulation(vw, num_iterations, users, times, actions, get_cost)

plot_scr(num_iterations, scr)
# Instantiate learner in VW
vw = pyvw.vw("--cb_explore_adf -q UA --quiet --epsilon 0.2")

num_iterations = 5000
scr = run_simulation(vw, num_iterations, users, times, actions, get_cost, do_learn=False)

plot_scr(num_iterations, scr)
def get_cost_new1(context,action):
    if context['user'] == "Client_type_1":
        if context['quarter'] == "2nd" and action == 'credit_large':
            return CUSTOMER_PURCHASED_PRODUCT
        elif context['quarter'] == "4th" and action == 'deposit_small':
            return CUSTOMER_PURCHASED_PRODUCT
        else:
            return CUSTOMER_DID_NOT_PURCHASE_PRODUCT
    elif context['user'] == "Client_type_2":
        if context['quarter'] == "2nd" and action == 'deposit_small':
            return CUSTOMER_PURCHASED_PRODUCT
        elif context['quarter'] == "4th" and action == 'deposit_small':
            return CUSTOMER_PURCHASED_PRODUCT
        else:
            return CUSTOMER_DID_NOT_PURCHASE_PRODUCT
        
get_preference_matrix(get_cost_new1)
def run_simulation_multiple_cost_functions(vw, num_iterations, users, times, actions, cost_functions, do_learn = True):
    cost_sum = 0.
    scr = []

    start_counter = 1
    end_counter = start_counter + num_iterations
    for cost_function in cost_functions:
        for i in range(start_counter, end_counter):
            # 1. in each simulation choose a user
            user = choose_user(users)
            # 2. choose time of day for a given user
            time = choose_time(times)

            # Construct context based on chosen user and time of day
            context = {'user': user, 'quarter': time}

            # 3. Use the get_action function we defined earlier
            action, prob = get_action(vw, context, actions)

            # 4. Get cost of the action we chose
            cost = cost_function(context, action)
            cost_sum += cost

            if do_learn:
                # 5. Inform VW of what happened so we can learn from it
                vw_format = vw.parse(to_vw_example_format(context, actions, (action, cost, prob)),pyvw.vw.lContextualBandit)
                # 6. Learn
                vw.learn(vw_format)

            # We negate this so that on the plot instead of minimizing cost, we are maximizing reward
            scr.append(-1*cost_sum/i)
        start_counter = end_counter
        end_counter = start_counter + num_iterations

    return scr
# use first reward function initially and then switch to second reward function

# Instantiate learner in VW
vw = pyvw.vw("--cb_explore_adf -q UA --quiet --epsilon 0.2")

num_iterations_per_cost_func = 5000
cost_functions = [get_cost, get_cost_new1]
total_iterations = num_iterations_per_cost_func * len(cost_functions)

scr = run_simulation_multiple_cost_functions(vw, num_iterations_per_cost_func, users, times, actions, cost_functions)

plot_scr(total_iterations, scr)
# Do not learn
# use first reward function initially and then switch to second reward function

# Instantiate learner in VW
vw = pyvw.vw("--cb_explore_adf -q UA --quiet --epsilon 0.2")

num_iterations_per_cost_func = 5000
cost_functions = [get_cost, get_cost_new1]
total_iterations = num_iterations_per_cost_func * len(cost_functions)

scr = run_simulation_multiple_cost_functions(vw, num_iterations_per_cost_func, users, times, actions, cost_functions, do_learn=False)
plot_scr(total_iterations, scr)
def get_cost_new2(context,action):
    if context['user'] == "Client_type_1":
        if context['quarter'] == "2nd" and action == 'credit_large':
            return CUSTOMER_PURCHASED_PRODUCT
        elif context['quarter'] == "4th" and action == 'leasing_medium':
            return CUSTOMER_PURCHASED_PRODUCT
        else:
            return CUSTOMER_DID_NOT_PURCHASE_PRODUCT
    elif context['user'] == "Client_type_2":
        if context['quarter'] == "2nd" and action == 'leasing_medium':
            return CUSTOMER_PURCHASED_PRODUCT
        elif context['quarter'] == "4th" and action == 'leasing_medium':
            return CUSTOMER_PURCHASED_PRODUCT
        else:
            return CUSTOMER_DID_NOT_PURCHASE_PRODUCT
# use first reward function initially and then switch to third reward function

# Instantiate learner in VW
# vw = pyvw.vw("--cb_explore_adf -q UA --quiet --epsilon 0.2")
vw = pyvw.vw("--cb_explore_adf -q UA --quiet --epsilon 0.00")

num_iterations_per_cost_func = 5000
cost_functions = [get_cost, get_cost_new2]
total_iterations = num_iterations_per_cost_func * len(cost_functions)

scr = run_simulation_multiple_cost_functions(vw, num_iterations_per_cost_func, users, times, actions, cost_functions)

plot_scr(total_iterations, scr)
# Do not learn
# use first reward function initially and then switch to third reward function

# Instantiate learner in VW
vw = pyvw.vw("--cb_explore_adf -q UA --quiet --epsilon 0.2")

num_iterations_per_cost_func = 5000
cost_functions = [get_cost, get_cost_new2]
total_iterations = num_iterations_per_cost_func * len(cost_functions)

scr = run_simulation_multiple_cost_functions(vw, num_iterations_per_cost_func, users, times, actions, cost_functions, do_learn=False)

plot_scr(total_iterations, scr)