import sys

sys.path.append('../input/flight-revenue-simulator/')

from flight_revenue_simulator import simulate_revenue, score_me

print('ok')
# This is my machine learning version of this challenge

# Try to do RL from scratch similar to Krapathy Pong from Pixels



import sys

import numpy as np

import pickle as pickle



sys.path.append('../input/flight-revenue-simulator/')

from flight_revenue_simulator import simulate_revenue, score_me



D         = 3     # Dimensions of input

H         = 92    # Number of hidden nodes

NORMAL    = 200.0 # Normalization Variable

RB        = 175.0 # Baseline reward ## Was 100 --- what is the right number????



MAX_PRICE  = 200

f_training = False

resume     = True



# Build or load the model

if resume:

    model = pickle.load(open('../input/50runs/save.p', 'rb'))

else:

    model = {}

    model['W1'] = np.random.randn(H,D) / np.sqrt(D) # Xavier Initialization

    model['W2'] = np.random.randn(H) / np.sqrt(H)



grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # Gradient update buffers to add

rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rms prop memory



def sigmoid(x):

    return 1.0 / (1.0 + np.exp(-x))



def policy_fwd(x):

    h = np.dot(model['W1'], x) # hidden layer

    h[h<0] = 0 # Relu -- consider changing the non-linearity

    logp = np.dot(model['W2'], h)

    p = sigmoid(logp) # output non-linearity --> probability

    return p, h



def policy_backward():

    # uses all global variables 

    global eph, epdlogp, epx, model

    

    dW2 = np.dot(eph.T, epdlogp).ravel() # backprop through output layer -- does this ignore the sigmoid?

    dh = np.outer(epdlogp, model['W2'])

    dh[eph<=0] = 0

    dW1 = np.dot(dh.T, epx)

    return {'W1':dW1, 'W2':dW2}



def pricing_function(days_left, tickets_left, demand_level):

    # use the training flag to know if we are training or now

    # if we are training, sample a space near the desired output; otherwise, get model prediction

    global f_training, xs, hs, days_to_sell,dlogps

    # Generate x with proper normalization to keep matrices happy

    x = [days_left/NORMAL, tickets_left/NORMAL, demand_level/NORMAL]

    if(f_training):

        p, h = policy_fwd(x)

        xs.append(x) # store input states

        hs.append(h) # store hidden states

        days_to_sell += 1  # This is really counting the number of states

        y = np.random.normal(p,0.05) # generate a random number near desired output of the model

                                     # This is designed to help explore the space

        # The following is from Karpathy code, it may or may not make sense for this application since this

        # not a classifier.  

        # Quick exploration of y-p function

        #     y is a scaled version of actual price set

        #     p is the target price that the model is suggesting

        # y-p approaches zero if the sampeled number is the target of the network (no updates will happen)

        # y-p > 0 if actual price is greater thant he model target price

        # y-p < 0 if the model target price is greater than the actual price

        #

        # TODO: Come back later and make sure the signs make sense, then possibly re-evaluate if this is

        #       the right function to use        

        dlogps.append(y-p)

        price = y * MAX_PRICE

    else:

        p, _ = policy_fwd(x)

        price = p * MAX_PRICE

    return price



def model_test(n):

    # n - how many times to run before averaging

    

    # store current value of the training flag

    global f_training

    f_t = f_training

    f_training = False

    

    # Variables to store total tickets sold and dollars made

    t_total = 0

    s_total = 0

    

    # iterate through forward prop

    for i in range(n):

        d = np.random.randint(1,high=NORMAL) # Generate a random number of days

        t = np.random.randint(1,high=NORMAL) # Generate a random number of tickets remaining    

        s = simulate_revenue(days_left=d, tickets_left=t, pricing_function=pricing_function, verbose=False)

        

        s_total += s

        t_total += t

        

    average_ticket_price = (1.0*s_total)/(1.0*t_total)

    

    # reset the training flag

    f_training = f_t

    return average_ticket_price



def model_train():

    global f_training

    f_t = f_training

    f_training = True

    

    # TODO: Consider taking samples from randome distributions biased towards 0

    

    d = np.random.randint(1,high=NORMAL) # Generate a random number of days

    t = np.random.randint(1,high=NORMAL) # Generate a random number of tickets remaining

        

    # Get a total revenue for random input vector x

    a = simulate_revenue(days_left=d, tickets_left=t, pricing_function=pricing_function, verbose=False)

        

    # Intial try at reward fuction

    # Calculate the average ticket price based on total revenue

    # This is the value we will try to maximize

    

    # Initially, we will assign global reward per 'simulate_revenue' and apply it evenly to every day

    # TODO: Consider outputing actual daily reward and using that in the pricing funciton if this doesn't work

    r = (a/t-RB)/RB

    

    f_training = f_t

    return r





## Start of code

batches = 50

batch_size = 10

games_per_batch = 5000

#learning_rate_max = 0.01

learning_rate = 0.005

decay_rate = 0.99

episode_number = 0

batch_counter = 50

model_check_iterations = 200



print('Setting a baseline...')

score_me(pricing_function)



for b in range(batches*batch_size):



    xs,hs,rs,dlogps = [],[],[],[]

    

    # At the beginning of each set of experiments, get a baseline of model

    average_ticket_price = model_test(model_check_iterations)

    print('MODEL TEST: --> Average Ticket Price over %d iterations is: $%0.2f' % (model_check_iterations, average_ticket_price))



    for g in range(games_per_batch):

        days_to_sell = 0 # need a counter to count the number of states output by the model

        r = model_train()

        d = days_to_sell



        for j in range(d):

            rs.append(r)



    episode_number += 1

    

    # Stack the states for easy math

    epr = np.vstack(rs)

    eph = np.vstack(hs)

    epx = np.vstack(xs)

    epdlogp = np.vstack(dlogps)



    epdlogp *= epr  # Multiply dlogp * reward function (PG Magic???)

    

    grad = policy_backward()

    for k in model: grad_buffer[k] += grad[k]

    

    #lrate = learning_rate_min+np.exp(-batch_counter/25.0)*learning_rate_max #exponential decay of learning rate

    

    # Training math from Karpathy directly

    if(episode_number % batch_size == 0):

        for k,v in model.items():

            g = grad_buffer[k]

            rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1- decay_rate) * g**2

            model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k])+1e-5) # gradient ascent

            grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

            

        batch_counter += 1

        print('Batch %d complete: Updating Model with a learning rate of %f' % (batch_counter, learning_rate))

        score_me(pricing_function)

        pickle.dump(model, open(('save.p.%d' % batch_counter), 'wb')) # 



#score_me(pricing_function)



pickle.dump(model, open('save.p', 'wb'))