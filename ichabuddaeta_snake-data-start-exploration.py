import pandas as pd

import matplotlib.pyplot as plt

import math
df = pd.read_csv('../input/snake-data/snakeData.txt')
df.shape
df.columns
def snake_to_food_distance(snakeX, snakeY, foodX, foodY):

    return math.sqrt((snakeX-foodX)**2+(snakeY-foodY)**2)
dist_to_food = []

for i, r in df.iterrows():

    dist_to_food.append(snake_to_food_distance(r.SnakeHeadX, r.SnakeHeadY, r.FoodX, r.FoodY))
df['dist_to_food'] = dist_to_food
df.dist_to_food.mean()
max(df.SnakeHeadY)
def head_to_corners_distance(snakeHeadX, snakeHeadY):

    topRightX, topRightY = (26,0)

    topLeftX, topLeftY = (0,0)

    bottomRightX, bottomRightY = (26,26)

    bottomLeftX, bottomLeftY = (0,26)

    dist_to_topRight = math.sqrt((snakeHeadX-topRightX)**2+(snakeHeadY-topRightY)**2)

    dist_to_bottomLeft = math.sqrt((snakeHeadX-bottomLeftX)**2+(snakeHeadY-bottomLeftY)**2)    

    dist_to_topLeft = math.sqrt((snakeHeadX-topLeftX)**2+(snakeHeadY-topLeftY)**2)    

    dist_to_bottomRight = math.sqrt((snakeHeadX-bottomRightX)**2+(snakeHeadY-bottomRightY)**2)

    return [dist_to_bottomLeft, dist_to_bottomRight, dist_to_topLeft, dist_to_topRight]
dist_topLeft = []

dist_topRight = []

dist_bottomLeft = []

dist_bottomRight = []

for i,r in df.iterrows():

    temp = head_to_corners_distance(r.SnakeHeadX,r.SnakeHeadY)

    dist_topLeft.append(temp[2])

    dist_topRight.append(temp[3])

    dist_bottomLeft.append(temp[0])

    dist_bottomRight.append(temp[1])
df['dist_to_topLeft'] = dist_topLeft

df['dist_to_topRight'] = dist_topRight

df['dist_to_bottomRight'] = dist_bottomRight

df['dist_to_bottomLeft'] = dist_bottomLeft
df_one = df[df.GameID==df.GameID[0]]
df_one.shape
plt.scatter(x=df_one.SnakeHeadX, y=df_one.SnakeHeadY)

plt.scatter(x=df_one.FoodX,y=df_one.FoodY)
plt.plot(df_one.dist_to_food)

plt.show()

plt.plot(df_one.dist_to_bottomLeft)

plt.show()

plt.plot(df_one.dist_to_bottomRight)

plt.show()

plt.plot(df_one.dist_to_topLeft)

plt.show()

plt.plot(df_one.dist_to_topRight)

df.GameID.count()
def get_game_plots(df,gameID):

    df_one = df[df.GameID==gameID].reset_index()

    print(f'Score of Game: {df_one.Score.max()}\nTotal Moves: {df_one.GameID.count()}\nScore to Moves Ratio: {round(df_one.Score.max()/df_one.GameID.count(),4)*100}%')

    plt.scatter(x=df_one.SnakeHeadX, y=df_one.SnakeHeadY)

    plt.scatter(x=df_one.FoodX,y=df_one.FoodY)

    plt.legend(labels=('Snake','Food'))

    plt.title(f'Snake from GameID {gameID}')

    plt.show()

    plt.plot(df_one.dist_to_food)

    plt.title('Moves vs. Distance to Food')

    plt.xlabel('Moves')

    plt.ylabel('Distance to Food')

    plt.show()

    plt.plot(df_one.dist_to_bottomLeft)

    plt.title('Moves vs. Distance to Bottom Left')

    plt.xlabel('Moves')

    plt.ylabel('Distance to Bottom Left')

    plt.show()

    plt.plot(df_one.dist_to_bottomRight)

    plt.title('Moves vs. Distance to Bottom Right')

    plt.xlabel('Moves')

    plt.ylabel('Distance to Bottom Right')

    plt.show()

    plt.plot(df_one.dist_to_topLeft)

    plt.title('Moves vs. Distance to Top Left')

    plt.xlabel('Moves')

    plt.ylabel('Distance to Top Left')

    plt.show()

    plt.plot(df_one.dist_to_topRight)

    plt.title('Moves vs. Distance to Top Right')

    plt.xlabel('Moves')

    plt.ylabel('Distance to Top Right')

    plt.show()

game_ids = list(df.GameID.drop_duplicates())

len(game_ids)
get_game_plots(df,game_ids[14])
get_game_plots(df, game_ids[1])