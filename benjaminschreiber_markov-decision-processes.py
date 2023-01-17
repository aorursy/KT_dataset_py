! pip install mdptoolbox-hiive
! pip install gym
! pip install pymdptoolbox
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import hiive.mdptoolbox 
import hiive.mdptoolbox.mdp
import hiive.mdptoolbox.example
import mdptoolbox, mdptoolbox.example
import gym
import matplotlib.pyplot as plt
import time

# P, R = hiive.mdptoolbox.example.forest(S=2000, p=.01)
# pim_test = hiive.mdptoolbox.mdp.ValueIteration(P, R, 0.999999, epsilon=0.01, max_iter=1000)
# pim_test.run()
# pim_test.run_stats
# forest_pi_mdp = mdptoolbox.mdp.PolicyIterationModified(P, R, 0.99999, epsilon=0.01, max_iter=10**6, skip_check=True)
# forest_pi_mdp.run()
# forest_pi_mdp.policy
# print("forest_pi_mdp.policy", forest_pi_mdp.policy)

def plot_simple_data(x_var, y_var, x_label, y_label, title, figure_size=(4,3)):
    plt.rcParams["figure.figsize"] = figure_size
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.plot(x_var, y_var, 'o-')
    plt.show()

def plot_data_legend(x_vars, x_label, all_y_vars, y_var_labels, y_label, title, y_bounds=None):
    colors = ['red','orange','black','green','blue','violet']
    plt.rcParams["figure.figsize"] = (4,3)

    i = 0
    for y_var in all_y_vars:
#         if i == 2: # don't plot when i = 1 for cv
#             x_vars = x_vars[1:]
        plt.plot(x_vars, y_var, 'o-', color=colors[i % 6], label=y_var_labels[i])
        i += 1
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if y_bounds != None:
        plt.ylim(y_bounds)
    leg = plt.legend()
    plt.show()

def make_time_array(run_stats, variables):
    cumulative_sum = 0
    times = []
    output_dict = {v:[] for v in variables}
    output_dict["times"] = times
    for result in run_stats:
        times.append(result["Time"])
        for v in result:
            if v in variables:
                output_dict[v].append(result[v])
    return output_dict
    
P, R = hiive.mdptoolbox.example.forest(S=2000, p=0.01)
st = time.time()
fm_q_mdp = hiive.mdptoolbox.mdp.QLearning(P, R, 0.999, epsilon=0.1,epsilon_decay=0.95, n_iter=1000000, alpha=0.95, skip_check=True)
fm_q_mdp.run()
end = time.time()
end-st
fm_q_mdp.policy
fm_q_curated_results = make_time_array(fm_q_mdp.run_stats, ["Mean V", "Max V", "Iteration"])
num_iters = len(fm_q_curated_results["Mean V"])
plot_simple_data(fm_q_curated_results["Iteration"], fm_q_curated_results["Mean V"], 
                 "iteration", "Mean Value", "Q-Learning Forest Mgmt Mean Value over Training", figure_size=(6,4))
plot_simple_data(fm_q_curated_results["Iteration"], fm_q_curated_results["Max V"], 
                 "iteration", "Max Value", "Q-Learning Forest Mgmt Max Value over Training", figure_size=(6,4))
plot_simple_data(fm_q_curated_results["Iteration"], fm_q_curated_results["times"], 
                 "iteration", "time elapsed (seconds)", "Q-Learning Forest Mgmt Time Elapsed over Training", figure_size=(6,4))
def compose_discounts(significant_digits):
    prev_discount = 0
    discounts = []
    for i in range(1,significant_digits + 1):
        discounts.append(round(prev_discount + 9*(10**-i),i))
        prev_discount = discounts[-1]
    return discounts

def run_forest(solver, states, discounts, epsilons, probability=0.1, max_iter=10):
    experiments = [] #num states, probability, discount, time, iterations, policy
    for s in states:
        for e in epsilons:
            for d in discounts:
                entry = {}
                P, R = hiive.mdptoolbox.example.forest(S=s, p=probability)
                #start_time = time.time()
                args = {"transitions":P, "reward":R, "gamma":d, "epsilon":e, "max_iter":max_iter, "skip_check":True}
                mdp = solver(args)
                mdp.run()
                #end_time = time.time()
                entry["time"] = mdp.time
                entry["iterations"] = mdp.iter
                entry["policy"] = mdp.policy
                entry["num_states"] = s
                entry["probability"] = probability
                entry["discount"] = d
                entry["epsilon"] = e
                entry["run_stats"] = mdp.run_stats
                experiments.append(entry)
    return experiments
       

#number of states by time/iterations
#discount over time/iterations
#p over number of 1's in policy
#epsilon over quality

#TODO quality
states = [10**s for s in range(1,4)]
discounts = compose_discounts(3)
discounts = [0.999999,0.9999999]
epsilons = [0.01, 0.005, 0.001]

fm_policy_iteration = lambda dict_args: hiive.mdptoolbox.mdp.PolicyIterationModified(**dict_args)
fm_policy_iteration_results = run_forest(fm_policy_iteration, states, discounts, epsilons)

states = [10**s for s in range(2,4)]
discounts = compose_discounts(5)
epsilons = [0.01, 0.005, 0.001]


fm_value_iteration = lambda dict_args: hiive.mdptoolbox.mdp.ValueIteration(**dict_args)
fm_value_iteration_results = run_forest(fm_value_iteration, states, discounts, epsilons)

fm_value_iteration_results
def print_training_results(results):
    for result in fm_value_iteration_results:
        print("\nNew result #################")
        for key in result:
            if key != "policy":
                print("{0}: {1}".format(key,result[key]))
def collect_training_results(results, keys, to_print=True):
    output_dict = {key:[] for key in keys}
    for result in results:
        if to_print: print("\nNew result #################")
        for key in result:
            if key in keys:
                if to_print: print("{0}: {1}".format(key,result[key]))
                output_dict[key].append(result[key])
    return output_dict
                

states = [k*10**pwr for pwr in range(2,3) for k in range(1,10)]
states += [1000]
states += [1000 + s for s in (states) ]

discounts = [0.9999999]
epsilons = [0.1]



fm_value_iteration = lambda dict_args: hiive.mdptoolbox.mdp.ValueIteration(**dict_args)
fm_value_iteration_results = run_forest(fm_value_iteration, states, discounts, epsilons)  
#print_training_results(fm_value_iteration_results)
fm_vi_time_num_states = collect_training_results(fm_value_iteration_results, ["time", "num_states"], to_print=False)
fm_vi_iters_num_states = collect_training_results(fm_value_iteration_results, ["iterations", "num_states"], to_print=False)
plot_simple_data(fm_vi_time_num_states["num_states"], fm_vi_time_num_states["time"], "num_states", "time", "Forest Mgmt Performance with Value Iteration")
fm_policy_iteration = lambda dict_args: hiive.mdptoolbox.mdp.PolicyIterationModified(**dict_args)
fm_policy_iteration_results = run_forest(fm_policy_iteration, states, discounts, epsilons)
fm_pi_time_num_states = collect_training_results(fm_policy_iteration_results, ["time", "num_states"], to_print=False)
fm_pi_iters_num_states = collect_training_results(fm_policy_iteration_results, ["iterations", "num_states"], to_print=False)
plot_simple_data(fm_pi_time_num_states["num_states"], fm_pi_time_num_states["time"], "num_states", "time", "Forest Mgmt Performance with Policy Iteration")

states = [2000]
discounts = [0.99]
epsilons = [0.75,0.5,0.25,0.1,0.01, 0.001]
fm_value_iteration = lambda dict_args: hiive.mdptoolbox.mdp.ValueIteration(**dict_args)
fm_value_iteration_results = run_forest(fm_value_iteration, states, discounts, epsilons, probability=0.0001, max_iter=10**2)  
#print_training_results(fm_value_iteration_results)
fm_vi_time_num_states = collect_training_results(fm_value_iteration_results, ["time", "epsilon", "iterations"], to_print=False)
plot_simple_data(fm_vi_time_num_states["epsilon"], fm_vi_time_num_states["time"], "epsilon", "time", "Forest Mgmt Value Iteration Training Time over Epsilon")
plot_simple_data(fm_vi_time_num_states["epsilon"], fm_vi_time_num_states["iterations"], "epsilon", "iterations", "Forest Mgmt Value Iteration Iterations over Epsilon")

states = [2000]
discounts = [0.99]
epsilons = [0.75,0.5,0.25,0.1,0.01, 0.001]
fm_policy_iteration = lambda dict_args: hiive.mdptoolbox.mdp.PolicyIterationModified(**dict_args)
fm_policy_iteration_results = run_forest(fm_policy_iteration, states, discounts, epsilons, probability=0.0001, max_iter=10**2)
fm_pi_time_num_states = collect_training_results(fm_policy_iteration_results, ["time", "epsilon"], to_print=False)
fm_pi_iters_num_states = collect_training_results(fm_policy_iteration_results, ["iterations", "epsilon"], to_print=False)
plot_simple_data(fm_pi_time_num_states["epsilon"], fm_pi_time_num_states["time"], "epsilon", "time", "Forest Mgmt Performance with Policy Iteration")
plot_simple_data(fm_pi_iters_num_states["epsilon"], fm_pi_iters_num_states["iterations"], "epsilon", "iterations", "Forest Mgmt Policy Iteration Iterations over Epsilon")

P_pi_fm, R_pi_fm = hiive.mdptoolbox.example.forest(S=2000, p=0.01)
dict_args = {"transitions":P_pi_fm, "reward":R_pi_fm, "gamma":0.9999, "epsilon":0.1, "max_iter":10**3}
fm_pi_mdp = hiive.mdptoolbox.mdp.PolicyIteration(P_pi_fm, R_pi_fm, 0.999, max_iter = 5*10**2, skip_check=True)
fm_pi_mdp.run()
print(fm_pi_mdp)
fm_pi_mdp_curated_results = make_time_array(fm_pi_mdp.run_stats, ["Mean V", "Max V"])
num_iters = len(fm_pi_mdp_curated_results["Mean V"])
plot_simple_data([i for i in range(num_iters)], fm_pi_mdp_curated_results["Mean V"], 
                 "iteration", "Mean Value", "PI Forest Mgmt Mean Value over Training", figure_size=(6,4))
plot_simple_data([i for i in range(num_iters)], fm_pi_mdp_curated_results["Max V"], 
                 "iteration", "Max Value", "PI Forest Mgmt Max Value over Training", figure_size=(6,4))
plot_simple_data([i for i in range(num_iters)], fm_pi_mdp_curated_results["times"], 
                 "iteration", "time elapsed (seconds)", "PI Forest Mgmt Time Elapsed over Training", figure_size=(6,4))
P_pi_fm, R_pi_fm = hiive.mdptoolbox.example.forest(S=2000, p=0.01)
dict_args = {"transitions":P_pi_fm, "reward":R_pi_fm, "gamma":0.999,"epsilon":10**(-50), "max_iter":10**5, "skip_check":True}
fm_vi_mdp = hiive.mdptoolbox.mdp.ValueIteration(**dict_args)
fm_vi_mdp.run()
print(fm_vi_mdp)
fm_vi_mdp_curated_results = make_time_array(fm_vi_mdp.run_stats, ["Mean V", "Max V"])
num_iters = len(fm_vi_mdp_curated_results["Mean V"])
print("max mean v", max(fm_vi_mdp_curated_results["Mean V"]))
plot_simple_data([i for i in range(num_iters)], fm_vi_mdp_curated_results["Mean V"], 
                 "iteration", "Mean Value", "VI Forest Mgmt Mean Value over Training", figure_size=(6,4))
plot_simple_data([i for i in range(num_iters)], fm_vi_mdp_curated_results["Max V"], 
                 "iteration", "Max Value", "VI Forest Mgmt Max Value over Training", figure_size=(6,4))
plot_simple_data([i for i in range(num_iters)], fm_vi_mdp_curated_results["times"], 
                 "iteration", "time elapsed (seconds)", "VI Forest Mgmt Time Elapsed over Training", figure_size=(6,4))
print("max mean v pi", max(fm_pi_mdp_curated_results["Mean V"]))
neq = []
for i in range(len(fm_vi_mdp.policy)):
    if fm_vi_mdp.policy[i] != fm_pi_mdp.policy[i]:
        neq.append(i)
len(neq)
sum(fm_vi_mdp.policy) < sum(fm_pi_mdp.policy)
sum(fm_pi_mdp.policy)
neq

P_pi_fm, R_pi_fm = hiive.mdptoolbox.example.forest(S=2000, p=0.01, r1=1000000)
dict_args = {"transitions":P_pi_fm, "reward":R_pi_fm, "gamma":0.999,"epsilon":10**(-10), "max_iter":10**5, "skip_check":True}
fm_vi_mdp = hiive.mdptoolbox.mdp.ValueIteration(**dict_args)
fm_vi_mdp.run()
print(fm_vi_mdp)
fm_vi_mdp_curated_results = make_time_array(fm_vi_mdp.run_stats, ["Mean V", "Max V"])
num_iters = len(fm_vi_mdp_curated_results["Mean V"])
print("max mean v", max(fm_vi_mdp_curated_results["Mean V"]))
plot_simple_data([i for i in range(num_iters)], fm_vi_mdp_curated_results["Mean V"], 
                 "iteration", "Mean Value", "VI Forest Mgmt Mean Value over Training", figure_size=(6,4))
plot_simple_data([i for i in range(num_iters)], fm_vi_mdp_curated_results["Max V"], 
                 "iteration", "Max Value", "VI Forest Mgmt Max Value over Training", figure_size=(6,4))
plot_simple_data([i for i in range(num_iters)], fm_vi_mdp_curated_results["times"], 
                 "iteration", "time elapsed (seconds)", "VI Forest Mgmt Time Elapsed over Training", figure_size=(6,4))
print("max mean v pi", max(fm_pi_mdp_curated_results["Mean V"]))
neq = []
for i in range(len(fm_vi_mdp.policy)):
    if fm_vi_mdp.policy[i] != fm_pi_mdp.policy[i]:
        neq.append(i)
len(neq)
sum(fm_vi_mdp.policy)
env = gym.make("FrozenLake-v0")
env.reset()
#Credit Blake Wang CS7641 @709_f1
nA, nS = env.nA, env.nS
P_fl = np.zeros([nA, nS, nS])
R_fl = np.zeros([nS, nA])
for s in range(nS):
    for a in range(nA):
        transitions = env.P[s][a]
        for p_trans, next_s, reward, _ in transitions:
            P_fl[a,s,next_s] += p_trans
            R_fl[s,a] = reward
        P_fl[a,s,:] /= np.sum(P_fl[a,s,:])


# frozen_q_policy = policy_iteration(frozen_lake_env, gamma = 0.4)
# policy_q_score = evaluate_policy(frozen_lake_env, frozen_pi_policy, gamma, n=1000)
P, R = hiive.mdptoolbox.example.forest(S=2000, p=0.01)
st = time.time()
fm_q_mdp = hiive.mdptoolbox.mdp.QLearning(P, R, 0.999, epsilon=0, n_iter=10**7, alpha=0.95, skip_check=True)
fm_q_mdp.run()
end = time.time()
end-st
fm_q_mdp.policy
fm_q_mdp.epsilon_decay
fm_q_curated_results = make_time_array(fm_q_mdp.run_stats, ["Mean V", "Max V", "Iteration"])
plot_simple_data(fm_q_curated_results["Iteration"], fm_q_curated_results["Mean V"], 
                 "iteration", "Mean Value", "Q-Learning Forest Mgmt Mean Value over Training", figure_size=(6,4))
plot_simple_data(fm_q_curated_results["Iteration"], fm_q_curated_results["Max V"], 
                 "iteration", "Max Value", "Q-Learning Forest Mgmt Max Value over Training", figure_size=(6,4))
plot_simple_data(fm_q_curated_results["Iteration"], fm_q_curated_results["times"], 
                 "iteration", "time elapsed (seconds)", "Q-Learning Forest Mgmt Time Elapsed over Training", figure_size=(6,4))
fl_q_mdp = hiive.mdptoolbox.mdp.QLearning(P_fl, R_fl, 0.99, epsilon=0.0,epsilon_decay=.95, n_iter=10**7, alpha=0.95, skip_check=True)
fl_q_mdp.run()
fm_q_mdp.policy
fl_q_curated_results = make_time_array(fl_q_mdp.run_stats, ["Mean V", "Max V", "Iteration"])
plot_simple_data(fl_q_curated_results["Iteration"], fl_q_curated_results["Mean V"], 
                 "iteration", "Mean Value", "Q-Learning Frozen Lake Mean Value over Training", figure_size=(6,4))
plot_simple_data(fl_q_curated_results["Iteration"], fl_q_curated_results["Max V"], 
                 "iteration", "Max Value", "Q-Learning Frozen Lake Max Value over Training", figure_size=(6,4))
plot_simple_data(fl_q_curated_results["Iteration"], fl_q_curated_results["times"], 
                 "iteration", "time elapsed (seconds)", "Q-Learning Frozen Lake Time Elapsed over Training", figure_size=(6,4))
dict_args = {"transitions":P_fl, "reward":R_fl, "gamma":0.999,"epsilon":10**(-10), "max_iter":10**5, "skip_check":True}
fm_vi_mdp = hiive.mdptoolbox.mdp.ValueIteration(**dict_args)
fm_vi_mdp.run()
#print(fm_vi_mdp)
fm_vi_mdp_curated_results = make_time_array(fm_vi_mdp.run_stats, ["Mean V", "Max V", "Iteration"])
num_iters = len(fm_vi_mdp_curated_results["Mean V"])
print("max mean v", max(fm_vi_mdp_curated_results["Mean V"]))
plot_simple_data(fm_vi_mdp_curated_results["Iteration"], fm_vi_mdp_curated_results["Mean V"], 
                 "iteration", "Mean Value", "VI Forest Mgmt Mean Value over Training", figure_size=(6,4))
plot_simple_data(fm_vi_mdp_curated_results["Iteration"], fm_vi_mdp_curated_results["Max V"], 
                 "iteration", "Max Value", "VI Forest Mgmt Max Value over Training", figure_size=(6,4))
plot_simple_data(fm_vi_mdp_curated_results["Iteration"], fm_vi_mdp_curated_results["times"], 
                 "iteration", "time elapsed (seconds)", "VI Forest Mgmt Time Elapsed over Training", figure_size=(6,4))
dict_args = {"transitions":P_fl, "reward":R_fl, "gamma":0.999,"epsilon":10**(-10), "max_iter":10**5, "skip_check":True}
fm_vi_mdp = hiive.mdptoolbox.mdp.ValueIteration(**dict_args)
fm_vi_mdp.run()
#print(fm_vi_mdp)
fm_vi_mdp_curated_results = make_time_array(fm_vi_mdp.run_stats, ["Mean V", "Max V", "Iteration"])
num_iters = len(fm_vi_mdp_curated_results["Mean V"])
print("max mean v", max(fm_vi_mdp_curated_results["Mean V"]))
plot_simple_data(fm_vi_mdp_curated_results["Iteration"], fm_vi_mdp_curated_results["Mean V"], 
                 "iteration", "Mean Value", "VI Forest Mgmt Mean Value over Training", figure_size=(6,4))
plot_simple_data(fm_vi_mdp_curated_results["Iteration"], fm_vi_mdp_curated_results["Max V"], 
                 "iteration", "Max Value", "VI Forest Mgmt Max Value over Training", figure_size=(6,4))
plot_simple_data(fm_vi_mdp_curated_results["Iteration"], fm_vi_mdp_curated_results["times"], 
                 "iteration", "time elapsed (seconds)", "VI Forest Mgmt Time Elapsed over Training", figure_size=(6,4))


# frozen_q_policy = policy_iteration(frozen_lake_env, gamma = 0.4)
# policy_q_score = evaluate_policy(frozen_lake_env, frozen_pi_policy, gamma, n=1000)