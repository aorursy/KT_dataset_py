import json

from pathlib import Path



data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')

train_path = data_path / 'training'

eval_path = data_path / 'evaluation'

test_path = data_path / 'test'



train_tasks = { task.stem: json.load(task.open()) for task in train_path.iterdir() } 

valid_tasks = { task.stem: json.load(task.open()) for task in eval_path.iterdir() }

eval_tasks = { task.stem: json.load(task.open()) for task in eval_path.iterdir() }



# Correct wrong cases:

# 025d127b

for i in range(9, 12):

    for j in range(3, 8):

        train_tasks['025d127b']['train'][0]['output'][i][j] = 0

for i in range(7, 10):

    for j in range(3, 6):

        train_tasks['025d127b']['train'][0]['output'][i][j] = 2

train_tasks['025d127b']['train'][0]['output'][8][4] = 0

# ef135b50

train_tasks['ef135b50']['test'][0]['output'][6][4] = 9

# bd14c3bf

for i in range(3):

    for j in range(5):

        if valid_tasks['bd14c3bf']['test'][0]['input'][i][j] == 1:

            valid_tasks['bd14c3bf']['test'][0]['input'][i][j] = 2

# a8610ef7

for i in range(6):

    for j in range(6):

        if valid_tasks['a8610ef7']['test'][0]['output'][i][j] == 8:

            valid_tasks['a8610ef7']['test'][0]['output'][i][j] = 5

valid_tasks['a8610ef7']['train'][3]['input'][0][1] = 2

valid_tasks['a8610ef7']['train'][3]['input'][5][1] = 2

# 54db823b

valid_tasks['54db823b']['train'][0]['output'][2][3] = 3

valid_tasks['54db823b']['train'][0]['output'][2][4] = 9

# e5062a87

for j in range(3, 7):

    train_tasks['e5062a87']['train'][1]['output'][1][j] = 2

# 1b60fb0c

train_tasks['1b60fb0c']['train'][1]['output'][8][8] = 0

train_tasks['1b60fb0c']['train'][1]['output'][8][9] = 0

# 82819916

train_tasks['82819916']['train'][0]['output'][4][5] = 4

# fea12743

for i in range(11, 16):

    for j in range(6):

        if valid_tasks['fea12743']['train'][0]['output'][i][j] == 2:

            valid_tasks['fea12743']['train'][0]['output'][i][j] = 8

# 42a50994

train_tasks['42a50994']['train'][0]['output'][1][0] = 8

train_tasks['42a50994']['train'][0]['output'][0][1] = 8

# f8be4b64

for j in range(19):

    if valid_tasks['f8be4b64']['test'][0]['output'][12][j] == 0:

        valid_tasks['f8be4b64']['test'][0]['output'][12][j] = 1

valid_tasks['f8be4b64']['test'][0]['output'][12][8] = 0

# d511f180

train_tasks['d511f180']['train'][1]['output'][2][2] = 9

# 10fcaaa3

train_tasks['10fcaaa3']['train'][1]['output'][4][7] = 8

# cbded52d

train_tasks['cbded52d']['train'][0]['input'][4][6] = 1

# 11852cab

train_tasks['11852cab']['train'][0]['input'][1][2] = 3

# 868de0fa

for j in range(2, 9):

    train_tasks['868de0fa']['train'][2]['input'][9][j] = 0

    train_tasks['868de0fa']['train'][2]['input'][10][j] = 1

    train_tasks['868de0fa']['train'][2]['input'][15][j] = 0

    train_tasks['868de0fa']['train'][2]['input'][16][j] = 1

train_tasks['868de0fa']['train'][2]['input'][15][2] = 1

train_tasks['868de0fa']['train'][2]['input'][15][8] = 1

# 6d58a25d

train_tasks['6d58a25d']['train'][0]['output'][10][0] = 0

train_tasks['6d58a25d']['train'][2]['output'][6][13] = 4

# a9f96cdd

train_tasks['a9f96cdd']['train'][3]['output'][1][3] = 0

# 48131b3c

valid_tasks['48131b3c']['train'][2]['output'][4][4] = 0

# 150deff5

aux = train_tasks['150deff5']['train'][2]['output'].copy()

train_tasks['150deff5']['train'][2]['output'] = train_tasks['150deff5']['train'][2]['input'].copy()

train_tasks['150deff5']['train'][2]['input'] = aux

# 17cae0c1

for i in range(3):

    for j in range(3, 6):

        valid_tasks['17cae0c1']['test'][0]['output'][i][j] = 9

# e48d4e1a

train_tasks['e48d4e1a']['train'][3]['input'][0][9] = 5

train_tasks['e48d4e1a']['train'][3]['output'][0][9] = 0

# 8fbca751

valid_tasks['8fbca751']['train'][1]['output'][1][3] = 2

valid_tasks['8fbca751']['train'][1]['output'][2][3] = 8

# 4938f0c2

for i in range(12):

    for j in range(6,13):

        if train_tasks['4938f0c2']['train'][2]['input'][i][j]==2:

            train_tasks['4938f0c2']['train'][2]['input'][i][j] = 0

for i in range(5,11):

    for j in range(7):

        if train_tasks['4938f0c2']['train'][2]['input'][i][j]==2:

            train_tasks['4938f0c2']['train'][2]['input'][i][j] = 0

# 9aec4887

train_tasks['9aec4887']['train'][0]['output'][1][4] = 8

# b0f4d537

for i in range(9):

    valid_tasks['b0f4d537']['train'][0]['output'][i][3] = 0

    valid_tasks['b0f4d537']['train'][0]['output'][i][4] = 1

valid_tasks['b0f4d537']['train'][0]['output'][2][3] = 3

valid_tasks['b0f4d537']['train'][0]['output'][2][4] = 3

valid_tasks['b0f4d537']['train'][0]['output'][5][3] = 2

# aa300dc3

valid_tasks['aa300dc3']['train'][1]['input'][1][7] = 5

valid_tasks['aa300dc3']['train'][1]['output'][1][7] = 5

valid_tasks['aa300dc3']['train'][1]['input'][8][2] = 5

valid_tasks['aa300dc3']['train'][1]['output'][8][2] = 5

# ad7e01d0

valid_tasks['ad7e01d0']['train'][0]['output'][6][7] = 0

# a8610ef7

valid_tasks['a8610ef7']['train'][3]['input'][0][1] = 0

valid_tasks['a8610ef7']['train'][3]['input'][5][1] = 0

valid_tasks['a8610ef7']['train'][3]['output'][0][1] = 0

valid_tasks['a8610ef7']['train'][3]['output'][5][1] = 0

# 97239e3d

valid_tasks['97239e3d']['test'][0]['input'][14][6] = 0

valid_tasks['97239e3d']['test'][0]['input'][14][10] = 0

# d687bc17

train_tasks['d687bc17']['train'][2]['output'][7][1] = 4