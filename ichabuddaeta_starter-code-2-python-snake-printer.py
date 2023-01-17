import json



snake_data = '../input/snake-data/snakeData - Copy.txt'
def get_snake_data(dataLine):

    snake_all = json.loads(dataLine.replace('\\"', '').split(',"')[-1].replace('\n','').replace('"','').replace("X",'"X"').replace("Y",'"Y"'))

    snake_head = {'X': int(dataLine.split(',')[3]), 'Y':int(dataLine.split(',')[4])}

    food_pos = {'X': int(dataLine.split(',')[5]), 'Y':int(dataLine.split(',')[6])}

    game_id = dataLine.split(',')[0]

    return {"Game_ID": game_id,'Snake_All': snake_all, 'Snake_Head': snake_head, "Food_Pos": food_pos}
snake_data_full = []

with open(snake_data, 'r') as snakeData:

    snakeData.readline()

    for line in snakeData.readlines():

        snake_data_full.append(get_snake_data(line))

    

    snakeData.close()

   
snake_data_full[0:5]
def print_snake_move(move_data = snake_data_full[1]):

    print(f'GAME ID: {move_data["Game_ID"]}')

    food_x = move_data['Food_Pos']['X']

    food_y = move_data['Food_Pos']['Y']

    snake_all_x = []

    snake_all_y = []

    for snake in move_data['Snake_All']:

        snake_all_x.append(snake['X'])

        snake_all_y.append(snake['Y'])

    print()

    print()

    lines = '|' + '-' * 80 + '|\n'

    for i in range(27):

        temp_line = ['|  ' for i in range(27)]

        if food_y == i:

            temp_line[food_x] = '|ff'

        for j in range(len(snake_all_y)):

            if snake_all_y[j] == i and j == 0:

                temp_line[snake_all_x[j]] = '|hh'

            elif snake_all_y[j] == i and j != 0:

                temp_line[snake_all_x[j]] = '|ss'

        lines +=(''.join(temp_line) + '|\n' + '|' + '-' * 80 + '|\n')

    

    print(lines)

print_snake_move(snake_data_full[0])
print_snake_move(snake_data_full[1])
print_snake_move(snake_data_full[2])
print_snake_move(snake_data_full[3])
print_snake_move(snake_data_full[4])
print_snake_move(snake_data_full[2020])