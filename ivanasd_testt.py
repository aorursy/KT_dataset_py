import random
THRESSHOLD = 1

WEIGHT_REDUCER = 0.0101

WEIGHT_INCREMENTOR = 0.01

def run(input, state=([0.1, -0.1], 0, False), skip_weight_changes = False):    

    w, current_charge, off = state

    

    if off == True:

        if skip_weight_changes==False:

            w[0] = w[0]-WEIGHT_REDUCER*input[0]

            w[1] = w[1]-WEIGHT_REDUCER*input[1]

        return 0, (w, 0, False)

    

    #TODO: probably add some reducing of new_charge

    #TODO: add not linear function

    new_charge = current_charge + (input[0] * w[0] + input[1] * w[1])/2

    

    output = 0

    if new_charge >= THRESSHOLD:

        if skip_weight_changes==False:

            w[0] = w[0]+WEIGHT_INCREMENTOR*input[0]

            w[1] = w[1]+WEIGHT_INCREMENTOR*input[1]

        output = 1

        new_charge = 0

        off = True

        

    if new_charge<0:

        new_charge=0

    

    return output, (w, new_charge, off)

output, state = run([1,1])
for i in range(100000):

    val1 = random.randrange(2)

    val2 = random.randrange(2)

    output, state = run([val1,val2], state)

    output, state = run([val1,val2], state)

    output, state = run([val1,val2], state)

    output, state = run([val1,val2], state)

    output, state = run([val1,val2], state)

    output, state = run([val1,val2], state)

    output, state = run([val1,val2], state)

    output, state = run([val1,val2], state)

    output, state = run([val1,val2], state)
print(state)

print(run([0,0], state, True))

print(run([1,0], state, True))

print(run([0,1], state, True))

print(run([1,1], state, True))