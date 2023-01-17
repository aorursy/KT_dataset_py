def sense(p, sense_value, sensor_right, world):

    pHit = sensor_right

    pMiss = 1 - pHit

    q=[]

    for i in range(len(p)):

        row = []

        for j in range(len(p[i])):

            hit = (sense_value == world[i][j])

            row.append(p[i][j] * (hit * pHit + (1-hit) * pMiss))

        q.append(row)

    

    s = sum([sum(q[i]) for i in range(len(q))])

    for i in range(len(q)):

        for j in range(len(q[i])):

            q[i][j] = q[i][j] / s

    return q
def move(p, motion, p_move):

    q = []

    for i in range(len(p)):

        row = []

        for j in range(len(p[i])):

            s = p_move * p[(i-motion[0]) % len(p)][(j-motion[1]) % len(p[i])]

            s += (1-p_move)* p[i][j]

            row.append(s)

        q.append(row)

    

    return q
def localize(colors,measurements,motions,sensor_right,p_move):

    # initializes p to a uniform distribution over a grid of the same dimensions as colors

    pinit = 1.0 / (float(len(colors)) * float(len(colors[0])))

    p = [[pinit for row in range(len(colors[0]))] for col in range(len(colors))]

    

    for i in range(len(motions)):

        p = move(p, motions[i], p_move)

        p = sense(p, measurements[i], sensor_right, colors)

    

    return p
def show(p):

    rows = ['[' + ','.join(map(lambda x: '{0:.5f}'.format(x),r)) + ']' for r in p]

    print ('[' + ',\n '.join(rows) + ']')
#############################################################

# For the following test case, your output should be 

# [[0.01105, 0.02464, 0.06799, 0.04472, 0.02465],

#  [0.00715, 0.01017, 0.08696, 0.07988, 0.00935],

#  [0.00739, 0.00894, 0.11272, 0.35350, 0.04065],

#  [0.00910, 0.00715, 0.01434, 0.04313, 0.03642]]

# (within a tolerance of +/- 0.001 for each entry)



colors = [['R','G','G','R','R'],

          ['R','R','G','R','R'],

          ['R','R','G','G','R'],

          ['R','R','R','R','R']]

measurements = ['G','G','G','G','G']

motions = [[0,0],[0,1],[1,0],[1,0],[0,1]]

p = localize(colors,measurements,motions,sensor_right = 0.7, p_move = 0.8)

show(p) # displays your answer
colors = [['G', 'G', 'G'],

          ['G', 'R', 'G'],

          ['G', 'G', 'G']]

measurements = ['R']

motions = [[0,0]]

sensor_right = 1.0

p_move = 1.0

p = localize(colors,measurements,motions,sensor_right,p_move)

correct_answer = (

    [[0.0, 0.0, 0.0],

     [0.0, 1.0, 0.0],

     [0.0, 0.0, 0.0]])

show(p) # displays your answer
colors = [['G', 'G', 'G'],

          ['G', 'R', 'R'],

          ['G', 'G', 'G']]

measurements = ['R']

motions = [[0,0]]

sensor_right = 1.0

p_move = 1.0

p = localize(colors,measurements,motions,sensor_right,p_move)

correct_answer = (

    [[0.0, 0.0, 0.0],

     [0.0, 0.5, 0.5],

     [0.0, 0.0, 0.0]])

show(p) # displays your answer
colors = [['G', 'G', 'G'],

          ['G', 'R', 'R'],

          ['G', 'G', 'G']]

measurements = ['R']

motions = [[0,0]]

sensor_right = 0.8

p_move = 1.0

p = localize(colors,measurements,motions,sensor_right,p_move)

correct_answer = (

    [[0.06666666666, 0.06666666666, 0.06666666666],

     [0.06666666666, 0.26666666666, 0.26666666666],

     [0.06666666666, 0.06666666666, 0.06666666666]])

show(p) # displays your answer
colors = [['G', 'G', 'G'],

          ['G', 'R', 'R'],

          ['G', 'G', 'G']]

measurements = ['R', 'R']

motions = [[0,0], [0,1]]

sensor_right = 0.8

p_move = 1.0

p = localize(colors,measurements,motions,sensor_right,p_move)

correct_answer = (

    [[0.03333333333, 0.03333333333, 0.03333333333],

     [0.13333333333, 0.13333333333, 0.53333333333],

     [0.03333333333, 0.03333333333, 0.03333333333]])

show(p) # displays your answer
colors = [['G', 'G', 'G'],

          ['G', 'R', 'R'],

          ['G', 'G', 'G']]

measurements = ['R', 'R']

motions = [[0,0], [0,1]]

sensor_right = 1.0

p_move = 1.0

p = localize(colors,measurements,motions,sensor_right,p_move)

correct_answer = (

    [[0.0, 0.0, 0.0],

     [0.0, 0.0, 1.0],

     [0.0, 0.0, 0.0]])

show(p) # displays your answer
colors = [['G', 'G', 'G'],

          ['G', 'R', 'R'],

          ['G', 'G', 'G']]

measurements = ['R', 'R']

motions = [[0,0], [0,1]]

sensor_right = 0.8

p_move = 0.5

p = localize(colors,measurements,motions,sensor_right,p_move)

correct_answer = (

    [[0.0289855072, 0.0289855072, 0.0289855072],

     [0.0724637681, 0.2898550724, 0.4637681159],

     [0.0289855072, 0.0289855072, 0.0289855072]])

show(p) # displays your answer
colors = [['G', 'G', 'G'],

          ['G', 'R', 'R'],

          ['G', 'G', 'G']]

measurements = ['R', 'R']

motions = [[0,0], [0,1]]

sensor_right = 1.0

p_move = 0.5

p = localize(colors,measurements,motions,sensor_right,p_move)

correct_answer = (

    [[0.0, 0.0, 0.0],

     [0.0, 0.33333333, 0.66666666],

     [0.0, 0.0, 0.0]])

show(p) # displays your answer