knob_weight = 0.5
input_val = 0.5
goal_pred = 0.8

# think y = mx from slope-intercept formula (ignore the b or bias part for now)
pred = input_val * knob_weight
# squaring the error will always make it postive (even if pred - goal_pred is negative)
error = (pred - goal_pred) ** 2

# how much you missed by
print(error)
weight = 0.5
input_val = 0.5
goal_prediction = 0.8

step_amount = 0.001

for iteration in range(1101):
    prediction = input_val * weight
    error = (prediction - goal_prediction) ** 2
    print("Step:" + str(iteration) + " Error:" + str(error) + " Prediction:" + str(prediction))
    
    # after predicting the error, you add a step amount value to the weight
    # multiplied by the original input_val to get a new prediction
    # the up_error is the new error based on you adding the step_amount to the weight
    up_prediction = input_val * (weight + step_amount)
    up_error = (goal_prediction - up_prediction) ** 2
    
    # after predicting the error, you subtract a step amount value from the weight
    # multiplied by the original input_val to get a new prediction
    # the down_error is the new error based on you subtracting the step_amount from the weight
    down_prediction = input_val * (weight - step_amount)
    down_error = (goal_prediction - down_prediction) ** 2
    
    # finally you check to see which is bigger
    # based on the evaluation of the if statement, you assign the variable weight
    # which was declared at the top a new weight value
    if (down_error < up_error):
        weight = weight - step_amount
        
    if (down_error > up_error):
        weight = weight + step_amount
        
# the for loop is going to run 1101 times in trying to get the error as close to zero as possible
# with a better prediction and more accurate weight
weight = 0.5
goal_pred = 0.8
input_val = 0.5

for iteration in range(20):
    pred = input_val * weight
    error = (pred - goal_pred) ** 2
    # direction_and_amount is going to tell you which direction your new weight value should move
    # either postive or negative
    # and also scale the amount for you through multiplication (or stop it if input_val = 0)
    direction_and_amount = (pred - goal_pred) * input_val
    # your new weight value will be the current weight - the direction_and_amount
    weight = weight - direction_and_amount
    
    print("Step:" + str(iteration) + " Error:" + str(error) + " Prediction:" + str(pred))
weight, goal_pred, input_val = 0.0, 0.8, 0.5

for iteration in range(20):
    pred = input_val * weight
    # error - how far you are off from actual prediction, positive
    error = (pred - goal_pred) ** 2
    # delta - how far you are off from actual prediction
    delta = pred - goal_pred
    # weight_delta - how far you off from the actual prediction with proper scaling & negative reversal
    # with the actual input value
    weight_delta = delta * input_val
    # weight - modify weight accordingly to the weight_delta 
    # this is also decreasing the error (or getting it closer to 0) as well
    # since the new weight value is used to calculate the pred variable
    weight -= weight_delta
    
    print("Step:" + str(iteration) + " Error:" + str(error) + " Prediction:" + str(pred))
weight = 0.5
goal_pred = 0.8
input_val = 2
alpha = 0.1

for iteration in range(20):
    pred = input_val * weight
    error = (pred - goal_pred) ** 2
    # you're calculating your slope/derivative
    derivative = input_val * (pred - goal_pred)
    # because you want your derivative/slope to not overshoot with the new weight update
    # you want to scale it (through mulitplication) appropiately based on your input
    weight = weight - (alpha * derivative)
    
    print("Step:" + str(iteration) + " Error:" + str(error) + " Prediction:" + str(pred))