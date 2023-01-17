# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.patches as mpatches

import matplotlib.pyplot as plt

from datetime import datetime, timedelta, date

from statistics import mean 

import copy

from keras.models import Sequential

from keras.layers import LSTM, Dropout, Dense, Activation

from sklearn.preprocessing import StandardScaler

sns.set(rc={'figure.figsize':(11,8)})

pd.set_option('display.max_rows', 20)

pd.set_option('display.max_columns', 20)

pd.set_option('display.width', 1000)

plt.rcParams["axes.grid"] = False

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os





paths = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        paths.append(os.path.join(dirname,filename))



# Any results you write to the current directory are saved as output.
# General path variables which we'll use throughout the code

lotteryPath = paths[0]
# Any results you write to the current directory are saved as output.

lottery = pd.read_csv(lotteryPath, encoding='latin-1')

lottery
all_balls = {}

for i in range(1,7):

    ball_ser = lottery['Ball ' +str(i)].value_counts()

    for key in ball_ser.keys():

        all_balls[key] = all_balls.get(key,0) + ball_ser[key]

        

ball_ser = lottery['Extra Ball'].value_counts()

for key in ball_ser.keys():

    all_balls[key] = all_balls.get(key,0) + ball_ser[key]



all_balls = pd.Series(all_balls) 



plt.title('Distribution of all balls')

plt.xticks(rotation=0)

sns.barplot(x=all_balls.keys(), y=all_balls.values, palette="OrRd")
# Visualize the distributions of each ball

f, axes = plt.subplots(7, 1)

f.tight_layout() 

for i in range(1,7):

    ball_dist = lottery['Ball ' +str(i)].value_counts().sort_index()

    axes[i-1].set_title('Distribution of ball '+str(i))

    plt.xticks(rotation=90)

    sns.barplot(x=ball_dist.keys(), y=ball_dist.values, ax=axes[i-1], palette="PuBuGn_d")



ball_dist = lottery['Extra Ball'].value_counts().sort_index()

axes[6].set_title('Distribution of extra ball')

plt.xticks(rotation=90)

sns.barplot(x=ball_dist.keys(), y=ball_dist.values, ax=axes[6], palette="PuBuGn_d")
# Correlation matrix

def plotCorrelationMatrix(df, graphWidth):

    #filename = df.dataframeName

    df = df.dropna('columns') # drop columns with NaN

    del df['Draw Number']

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    if df.shape[1] < 2:

        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')

        return

    corr = df.corr('pearson')

    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1,cmap = "BuGn")

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=0)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

   

    plt.title(f'Correlation Matrix', fontsize=15)

    plt.show()

plotCorrelationMatrix(lottery, 8)
def getDate(strDate):

    return datetime.strptime(strDate, '%Y-%m-%d').date()
# This list will hold all of the monthly draws

allMonthsData = []



# The data that will be held will simply have the date of the draw along with the {"ball drawn": occurence}

class MonthData:

    def __init__(self,date,ballsDict):

        self.date = date

        self.ballsDict = ballsDict
# What we know about these draws is that they'll be part of the same month

def generateStatsForDraws(draws, drawDate):

    if(draws.empty == False):

        currentBalls = {}

        del draws['Date']

        del draws['Draw Number']

        del draws['Jackpot']

        balls_list = draws.values.T.tolist()

        balls_flat_list = [item for sublist in balls_list for item in sublist]

        for i in range(1,43):

            currentBalls[i] = balls_flat_list.count(i)

        data = MonthData(drawDate, currentBalls)

        allMonthsData.append(data)





        

def plotBallsInMonths(index):

    all_balls = pd.Series(allMonthsData[index].ballsDict) 

    plt.title('Balls in ' + str(allMonthsData[index].date.month) + "-" + str(allMonthsData[index].date.year))

    plt.xticks(rotation=0)

    sns.barplot(x=all_balls.keys(), y=all_balls.values, palette="GnBu_d")
ball_month = pd.DataFrame()



initDate = getDate(lottery['Date'][0])

currentMonth = initDate.month

currentYear = initDate.year



def getOccurencesPerMonth():

    global ball_month

    global currentMonth

    global currentYear

    for index, draw in lottery.iterrows():

        drawDate = getDate(draw['Date'])

        if(drawDate.month == currentMonth  and drawDate.year == currentYear): 

            ball_month = ball_month.append(draw)

        else:

            generateStatsForDraws(ball_month,drawDate)

            ball_month = pd.DataFrame()

            currentMonth = (currentMonth % 12) + 1 

            if(currentYear != drawDate.year):

                currentYear = drawDate.year

getOccurencesPerMonth()
plotBallsInMonths(50)
# I want to know on how many interval is each ball appearing

# For ex: #23 appeared X times this year



ball_dataset = pd.DataFrame(columns = ['Year', 'Ball Number', 'Occurences'])

ball_dataset["Year"] = pd.to_numeric(ball_dataset["Year"])

ball_dataset["Ball Number"] = pd.to_numeric(ball_dataset["Ball Number"])

ball_dataset["Occurences"] = pd.to_numeric(ball_dataset["Occurences"])







# What we know about these draws is that they'll be part of the same month

# So we can have a data frame of that month where we count the numbers

def generateYearStatsForDraws(draws, drawDate):

    global ball_dataset

    if(draws.empty == False):

        currentBalls = {}

        #print(draws)

        del draws['Date']

        #print(draws)

        balls_list = draws.values.T.tolist()

        balls_flat_list = [item for sublist in balls_list for item in sublist]

        #print(balls_flat_list)

        for i in range(1,43):

            currentBalls['Year'] = int(drawDate.year)

            currentBalls['Ball Number'] = int(i)

            currentBalls['Occurences'] = int(balls_flat_list.count(i))

            ball_at_year = pd.Series(currentBalls)

            currentBalls = {}

            ball_dataset = ball_dataset.append(ball_at_year, ignore_index = True)

        #print("yalla")

        

        

ball_month = pd.DataFrame()



initDate = getDate(lottery['Date'][0])

currentMonth = initDate.month

currentYear = initDate.year



for index, draw in lottery.iterrows():

    del draw['Draw Number']

    del draw['Jackpot']

    drawDate = getDate(draw['Date'])

    if(drawDate.month == currentMonth  and drawDate.year == currentYear): 

        #print(draw['Date'])

        ball_month = ball_month.append(draw)

    else:

        #break

        #print("New Month")

        currentMonth = (currentMonth % 12) + 1 

        if(currentYear != drawDate.year):

            #print("New Year")

            #print(ball_month)

            generateYearStatsForDraws(ball_month, drawDate)

            ball_month = pd.DataFrame()

            currentYear = drawDate.year





print(ball_dataset)    

    

        
balls = ball_dataset.pivot("Ball Number", "Year", "Occurences")

f, ax = plt.subplots(figsize=(18, 18))

plt.title("Occurence of Each Ball per Year")

sns.heatmap(balls, annot=True, fmt="d", linewidths=0.0, ax=ax)
# Get number of even and odd numbers in a given list

def getNumberOfEvenAndOdd(numbers):

    countEven = 0

    countOdd = 0

    for number in numbers:

        if number % 2 == 0:

            countEven += 1

        else:

            countOdd += 1

    return str(countEven) + " Even, " +str(countOdd) +" Odd"



def visualizeEvenOddCombination():

    even_odd = {}

    numbers = []

    for index, draw in lottery.iterrows():

            numbers.append(int(draw['Ball 1']))

            numbers.append(int(draw['Ball 2']))

            numbers.append(int(draw['Ball 3']))

            numbers.append(int(draw['Ball 4']))

            numbers.append(int(draw['Ball 5']))

            numbers.append(int(draw['Ball 6']))

            even_odd[getNumberOfEvenAndOdd(numbers)] = even_odd.get(getNumberOfEvenAndOdd(numbers), 0) + 1 

            numbers = []

    even_odd_ser = pd.Series(even_odd)

    print(even_odd_ser)

    # Pie chart, where the slices will be ordered and plotted counter-clockwise:

    labels = even_odd.keys()

    sizes = list(even_odd.values())

    sizes = [x / float(len(lottery)) * 100 for x in sizes]

    explode = (0.1, 0, 0, 0, 0,0,0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots(figsize=(15, 6))

    colors = ["#416fc4", "#48cfdf", "#48c4d6", "#41d6d3", "#5decd2", "#87fde8","#b4f6eb"]

    ax1.pie(sizes, explode=explode, labels=labels,colors = colors, shadow=True)

    plt.title("Even Odd Probabilities")

    plt.legend( loc = 'lower right', labels=['%s, %1.2f %%' % (l, s) for l, s in zip(labels, sizes)])

    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



    



visualizeEvenOddCombination()
# Get number of even and odd numbers in a given list

def getNumberOfHighAndLow(numbers):

    countHigh = 0

    countLow = 0

    for number in numbers:

        if number >= 22:

            countHigh += 1

        else:

            countLow += 1

    return str(countHigh) + " High, " +str(countLow) +" Low"



def visualizeHighLowCombination():

    high_low = {}

    numbers = []

    for index, draw in lottery.iterrows():

            numbers.append(int(draw['Ball 1']))

            numbers.append(int(draw['Ball 2']))

            numbers.append(int(draw['Ball 3']))

            numbers.append(int(draw['Ball 4']))

            numbers.append(int(draw['Ball 5']))

            numbers.append(int(draw['Ball 6']))

            high_low[getNumberOfHighAndLow(numbers)] = high_low.get(getNumberOfHighAndLow(numbers), 0) + 1 

            numbers = []

    high_low_ser = pd.Series(high_low)

    print(high_low_ser)

    # Pie chart, where the slices will be ordered and plotted counter-clockwise:

    labels = high_low.keys()

    sizes = list(high_low.values())

    sizes = [x / float(len(lottery)) * 100 for x in sizes]

    explode = (0.0, 0.1, 0, 0, 0,0,0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots(figsize=(15, 6))

    colors = ["#006F3F", "#006F4F", "#006F53", "#006753", "#007e65", "#0a9d80","#81c6b8"]

    ax1.pie(sizes, explode=explode, labels=labels,colors = colors, shadow=True)

    plt.title("High Low Probabilities")

    plt.legend( loc = 'lower right', labels=['%s, %1.2f %%' % (l, s) for l, s in zip(labels, sizes)])

    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



    



visualizeHighLowCombination()
def getMean(ball_dist):

    mean = 0

    allOcc = 0

    for value in list(ball_dist.keys()):

        for occ in list(ball_dist.values):

            mean += value * occ

            allOcc += occ

    mean = mean/allOcc

    return mean



sumOfMeans = 0

for i in range(1,7):

    ball_dist = lottery['Ball ' +str(i)].value_counts().sort_index()

    sumOfMeans += getMean(ball_dist)

sumOfMeans = int(sumOfMeans)

print(sumOfMeans)
# Get number of even and odd numbers in a given list

def isNumbersInRange(numbers):

    sumOfNumbers = 0 

    for number in numbers:

        sumOfNumbers += number

    if (sumOfNumbers >= sumOfMeans - 29 and sumOfNumbers <= sumOfMeans + 29):

        return "In Range"

    else:

        return "Out of Range"    





def visualizeInRangeCombination():

    in_range = {}

    numbers = []

    for index, draw in lottery.iterrows():

            numbers.append(int(draw['Ball 1']))

            numbers.append(int(draw['Ball 2']))

            numbers.append(int(draw['Ball 3']))

            numbers.append(int(draw['Ball 4']))

            numbers.append(int(draw['Ball 5']))

            numbers.append(int(draw['Ball 6']))

            in_range[isNumbersInRange(numbers)] = in_range.get(isNumbersInRange(numbers), 0) + 1 

            numbers = []

    in_range_ser = pd.Series(in_range)

    print(in_range_ser)

    # Pie chart, where the slices will be ordered and plotted counter-clockwise:

    labels = in_range.keys()

    sizes = list(in_range.values())

    sizes = [x / float(len(lottery)) * 100 for x in sizes]

    fig1, ax1 = plt.subplots(figsize=(15, 6))

    colors = ["#E4665C", "#F9B189"]

    ax1.pie(sizes, labels=labels, shadow=True, colors=colors)

    plt.title("Jackpot Sum between " + str(sumOfMeans - 29) + " and " + str(int(sumOfMeans + 29)))

    plt.legend( loc = 'lower right', labels=['%s, %1.2f %%' % (l, s) for l, s in zip(labels, sizes)])

    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



    



visualizeInRangeCombination()
# We'll assume that "isEliteNumbers(numbers)" function will determine that the numbers comform with the conditions specified above.



# I know, I can optimize my code!

def isEliteNumbers(numbers):

    for number in numbers:

        if (isNumbersInRange(numbers) == "In Range" and getNumberOfEvenAndOdd(numbers) == "3 Even, 3 Odd" and getNumberOfHighAndLow(numbers) == "3 High, 3 Low"):

            return "Elite Numbers"

        else:

            return "Other Numbers"    



    

def visualizeEliteCombination():

    numbers_type = {}

    numbers = []

    for index, draw in lottery.iterrows():

            numbers.append(int(draw['Ball 1']))

            numbers.append(int(draw['Ball 2']))

            numbers.append(int(draw['Ball 3']))

            numbers.append(int(draw['Ball 4']))

            numbers.append(int(draw['Ball 5']))

            numbers.append(int(draw['Ball 6']))

            numbers_type[isEliteNumbers(numbers)] = numbers_type.get(isEliteNumbers(numbers), 0) + 1 

            numbers = []

    numbers_type_ser = pd.Series(numbers_type)

    print(numbers_type_ser)

    # Pie chart, where the slices will be ordered and plotted counter-clockwise:

    labels = numbers_type.keys()

    sizes = list(numbers_type.values())

    sizes = [x / float(len(lottery)) * 100 for x in sizes]

    fig1, ax1 = plt.subplots(figsize=(15, 6))

    colors = ["#424F60", "#003041"]

    ax1.pie(sizes, labels=labels, shadow=True, colors=colors)

    plt.title("Elite Number probabilities")

    plt.legend( loc = 'lower right', labels=['%s, %1.2f %%' % (l, s) for l, s in zip(labels, sizes)])

    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



    



visualizeEliteCombination()
lottery_ml = copy.deepcopy(lottery)

del lottery_ml['Draw Number']

del lottery_ml['Jackpot']

del lottery_ml['Date']

lottery_ml.head()
# Normalizing and downscaling the data such that mean per column is 0

scaler = StandardScaler().fit(lottery_ml.values)

transformed_dataset = scaler.transform(lottery_ml.values)

lottery_ml_normalized = pd.DataFrame(data=transformed_dataset, index=lottery_ml.index)

lottery_ml_normalized
rows_to_retain_for_test = 350

number_of_rows= lottery_ml.values.shape[0] - rows_to_retain_for_test

games_window_size = 50 #amount of past games we need to take in consideration for training (It's also the number of draws)

number_of_features = lottery_ml.values.shape[1] #balls count
#Number of rows = number of games to train (samples)

#Number of columns = Number of previous games (timestep)

#Number of depths = Number of features (in this case on 7 balls) (features)

train = np.empty([number_of_rows-games_window_size, games_window_size, number_of_features], dtype=float)

label = np.empty([number_of_rows-games_window_size, number_of_features], dtype=float)





# train represents the training data

# label represents the expected result



#train[2][0][5] holds the value of ball 6 (or feature number 6) in previous game 0 (that means the current game) which is actually game number 1.

#train[200][4][3] holds the value of ball 4 (or feature number 4) in previous game 4 (that means 4 games before the current draw) where the current draw is 199.



#Think about it as a row based visualization



#For each row ==> Technically we can say that every row is a training batch



for i in range(0, number_of_rows-games_window_size):

    train[i]=lottery_ml_normalized.iloc[i:i+games_window_size, 0: number_of_features]

    # the label will have the result of the next draw.

    label[i]=lottery_ml_normalized.iloc[i+games_window_size: i+games_window_size + 1, 0: number_of_features]
model = Sequential()

model.add(LSTM(200,      

            input_shape=(games_window_size, number_of_features),

            return_sequences=True))

model.add(Dropout(0.2))

model.add(LSTM(200,           

               return_sequences=False))

model.add(Dropout(0.2))

model.add(Dense(number_of_features))

model.compile(loss='mse', optimizer='rmsprop', metrics=["accuracy"])



# train model normally

history =  model.fit(train, label, epochs= 500, validation_split=0.25, batch_size = 128, verbose=0)
# Plot training & validation accuracy values

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
validation_df_0 = pd.DataFrame(columns = ['Validation Draw','Ball','Value'])

validation_df_0["Validation Draw"] = pd.to_numeric(validation_df_0["Validation Draw"])

validation_df_0["Value"] = pd.to_numeric(validation_df_0["Value"])



validation_df_1 = pd.DataFrame(columns = ['Validation Draw','Ball','Value'])

validation_df_1["Validation Draw"] = pd.to_numeric(validation_df_1["Validation Draw"])

validation_df_1["Value"] = pd.to_numeric(validation_df_1["Value"])



validation_df_3 = pd.DataFrame(columns = ['Validation Draw','Ball','Value'])

validation_df_3["Validation Draw"] = pd.to_numeric(validation_df_3["Validation Draw"])

validation_df_3["Value"] = pd.to_numeric(validation_df_3["Value"])



validation_df_6 = pd.DataFrame(columns = ['Validation Draw','Ball','Value'])

validation_df_6["Validation Draw"] = pd.to_numeric(validation_df_6["Validation Draw"])

validation_df_6["Value"] = pd.to_numeric(validation_df_6["Value"])
for i in range(lottery_ml.values.shape[0]-rows_to_retain_for_test, lottery_ml.values.shape[0] - games_window_size ):

    to_predict = lottery_ml.iloc[i:i+games_window_size].values.tolist()

    scaled_to_predict = scaler.transform(to_predict)

    scaled_predicted_output_1 = model.predict(np.array([scaled_to_predict]))

    predicted_draw = scaler.inverse_transform(scaled_predicted_output_1).astype(int)[0]

    actual_draw = lottery_ml.iloc[i+games_window_size].values.tolist()

    validation_df_0 = validation_df_0.append({'Validation Draw': int(i), 'Ball': "Actual", 'Value': actual_draw[0]}, ignore_index=True)

    validation_df_0 = validation_df_0.append({'Validation Draw': int(i), 'Ball': "Predicted", 'Value': predicted_draw[0] }, ignore_index=True)

    validation_df_1 = validation_df_1.append({'Validation Draw': int(i), 'Ball': "Actual", 'Value': actual_draw[1]}, ignore_index=True)

    validation_df_1 = validation_df_1.append({'Validation Draw': int(i), 'Ball': "Predicted", 'Value': predicted_draw[1] }, ignore_index=True)

    validation_df_3 = validation_df_3.append({'Validation Draw': int(i), 'Ball': "Actual", 'Value': actual_draw[3]}, ignore_index=True)

    validation_df_3 = validation_df_3.append({'Validation Draw': int(i), 'Ball': "Predicted", 'Value': predicted_draw[3] }, ignore_index=True)

    validation_df_6 = validation_df_6.append({'Validation Draw': int(i), 'Ball': "Actual", 'Value': actual_draw[6]}, ignore_index=True)

    validation_df_6 = validation_df_6.append({'Validation Draw': int(i), 'Ball': "Predicted", 'Value': predicted_draw[6] }, ignore_index=True)
fig, ax = plt.subplots(figsize=(24, 8))

plt.title("Ball 1: Actual vs Predicted")

ax = sns.lineplot(x="Validation Draw", y="Value", hue="Ball",style="Ball",data=validation_df_0,markers=True, dashes=False)
fig, ax = plt.subplots(figsize=(24, 8))

plt.title("Ball 4: Actual vs Predicted")

ax = sns.lineplot(x="Validation Draw", y="Value", hue="Ball",style="Ball",data=validation_df_3,markers=True, dashes=False)
fig, ax = plt.subplots(figsize=(24, 8))

plt.title("Extra Ball: Actual vs Predicted")

ax = sns.lineplot(x="Validation Draw", y="Value", hue="Ball",style="Ball",data=validation_df_6,markers=True, dashes=False)


trained_df = pd.DataFrame(columns = ['Validation Draw','Ball','Value'])

trained_df["Validation Draw"] = pd.to_numeric(validation_df_0["Validation Draw"])

trained_df["Value"] = pd.to_numeric(trained_df["Value"])



for i in range(0, int((lottery_ml.values.shape[0]-rows_to_retain_for_test)/2)):

    to_predict = lottery_ml.iloc[i:i+games_window_size].values.tolist()

    scaled_to_predict = scaler.transform(to_predict)

    scaled_predicted_output_1 = model.predict(np.array([scaled_to_predict]))

    predicted_draw = scaler.inverse_transform(scaled_predicted_output_1).astype(int)[0]

    actual_draw = lottery_ml.iloc[i+games_window_size].values.tolist()

    trained_df = trained_df.append({'Validation Draw': int(i), 'Ball': "Actual", 'Value': actual_draw[0]}, ignore_index=True)

    trained_df = trained_df.append({'Validation Draw': int(i), 'Ball': "Predicted", 'Value': predicted_draw[0] }, ignore_index=True)

    

fig, ax = plt.subplots(figsize=(24, 8))

plt.title("Ball 1: Plot of prediction over trained data")

ax = sns.lineplot(x="Validation Draw", y="Value", hue="Ball",style="Ball",data=trained_df,markers=False, dashes=True)
from math import sqrt

from numpy import split

from numpy import array

from pandas import read_csv

from sklearn.metrics import mean_squared_error

from matplotlib import pyplot

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers import LSTM

from keras.layers import RepeatVector

from keras.layers import TimeDistributed

from keras.layers import ConvLSTM2D, Conv1D, MaxPooling1D

from keras.layers.normalization import BatchNormalization
def prepare_datasets(lottery_ml_normalized):

    train = np.empty([number_of_rows,games_window_size,number_of_features], dtype=int)

    label = np.empty([number_of_rows,number_of_features], dtype=int)

    test = np.empty([rows_to_retain_for_test,games_window_size,number_of_features], dtype=int)

    

    for i in range(0, number_of_rows):

        train[i]=lottery_ml_normalized.iloc[i:i+games_window_size, 0: number_of_features]

        label[i]=lottery_ml_normalized.iloc[i+games_window_size: i+games_window_size + 1, 0: number_of_features]



    for i in range(number_of_rows, number_of_rows + rows_to_retain_for_test - games_window_size):

        test[i-number_of_rows]=lottery_ml_normalized.iloc[i:i+games_window_size, 0: number_of_features]



    # current shape is [samples, timesteps, features]

    # reshape into [samples, subsequences, timesteps, features]



    # subsequences means the number of rows every single data represents.

    

    # In our case it is 1 subsequence,then the whole draw as treated as a single a row (that means the 6 balls)

    train = train.reshape([number_of_rows, 1, games_window_size, number_of_features])

    test = test.reshape([rows_to_retain_for_test, 1, games_window_size, number_of_features])

    

    return train, label, test
def build_model(train, label, n_epochs, val_ratio):

    

    model = Sequential()

    # input (batch, steps, channels)

    model.add(TimeDistributed(Conv1D(filters=3, kernel_size=16, activation='relu'), input_shape=(None, games_window_size, number_of_features)))

    model.add(TimeDistributed(Conv1D(filters=3, kernel_size=32, activation='relu')))

    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))

    model.add(TimeDistributed(Flatten()))

    model.add(LSTM(512, activation='relu',return_sequences=True))

    model.add(Dropout(0.5))

    model.add(LSTM(512, activation='relu'))

    model.add(Dropout(0.3))

    #model.add(Flatten())

    #model.add(LSTM(10, activation='relu'))

    model.add(Dense(7))



    model.compile(optimizer='adam', loss='mse', metrics=["accuracy"])



    history = model.fit(train, label,validation_split=val_ratio, epochs=n_epochs, verbose=0)

    

    return history
train, label, test = prepare_datasets(lottery_ml_normalized)

history = build_model(train, label, 2000, 0.3)
# Plot training & validation accuracy values

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
# As for the testing, we'll plot a confusion matrix

test_pred = pd.DataFrame(columns = ["Predicted", "Actual", "Occurences"])



test_pred["Predicted"] = pd.to_numeric(test_pred["Predicted"])

test_pred["Actual"] = pd.to_numeric(test_pred["Actual"])

test_pred["Occurences"] = pd.to_numeric(test_pred["Occurences"])





occurences = {}

#for test_in in test:

for i in range(0,test.shape[0]):

        prediction_normalized = model.predict(test[i], verbose = 0)

        pred = scaler.inverse_transform(prediction_normalized).astype(int)[0]

        actual = lottery_ml.iloc[i+rows_to_retain_for_test]

        for nb in range(0, 7):

            occurences[str(pred[nb]) + "" + str(actual[nb])]  = occurences.get(str(pred[nb]) + "" + str(actual[nb]), 0) + 1 

            if(occurences[str(pred[nb]) + "" + str(actual[nb])] == 1):

                test_pred = test_pred.append({'Predicted': pred[nb], 'Actual': actual[nb], 'Occurences': occurences[str(pred[nb]) + "" + str(actual[nb])] }, ignore_index=True)

            else:

                test_pred.loc[(test_pred["Predicted"] == pred[nb]) & (test_pred["Actual"] == actual[nb]), "Occurences"] = occurences.get(str(pred[nb]) + "" + str(actual[nb]), 0) + 1

                #print(ok)

        #print(pred)

        #print(actual)

        #break

print(test_pred)

test_pred_hm = test_pred.pivot("Actual", "Predicted", "Occurences")

f, ax = plt.subplots(figsize=(15, 15))

plt.title("Occurences of balls with their predicted result")

sns.heatmap(test_pred_hm, annot=True,linewidths=0.5, ax=ax)
lottery_ml_sum = copy.deepcopy(lottery_ml)

lottery_ml_sum['SUM'] = 0

lottery_ml_sum
def sumBalls(balls):

    sumCurrent = 0

    for ball in balls:

        sumCurrent += ball

    return sumCurrent



indexsToDrop = []

currentBalls = []

for index, draw in lottery_ml_sum.iterrows():

    currentBalls.append(draw['Ball 1'])

    currentBalls.append(draw['Ball 2'])

    currentBalls.append(draw['Ball 3'])

    currentBalls.append(draw['Ball 4'])

    currentBalls.append(draw['Ball 5'])

    currentBalls.append(draw['Ball 6'])

    sumCurrentBalls = sumBalls(currentBalls)

    if( sumCurrentBalls >= 100 and  sumCurrentBalls <= 158):

        draw['SUM'] = 100

    else:

        draw['SUM'] = 30

        indexsToDrop.append(index)  

    currentBalls = []



lottery_ml_sum = lottery_ml_sum.drop(indexsToDrop)



del lottery_ml_sum['SUM']

lottery_ml_sum
rows_to_retain_for_test = 350

number_of_rows= lottery_ml_sum.values.shape[0] - rows_to_retain_for_test

games_window_size = 50 #amount of past games we need to take in consideration for training (It's also the number of draws)

number_of_features = lottery_ml_sum.values.shape[1] #balls count
# Normalizing and downscaling the data such that mean per column is 0

scaler = StandardScaler().fit(lottery_ml_sum.values)

transformed_dataset = scaler.transform(lottery_ml_sum.values)

lottery_ml_normalized = pd.DataFrame(data=transformed_dataset, index=lottery_ml_sum.index)

number_of_features = lottery_ml_sum.values.shape[1] #balls count

lottery_ml_normalized
train, label, test = prepare_datasets(lottery_ml_normalized)

history = build_model(train, label, 500, 0.3)
# Plot training & validation accuracy values

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
lottery_ml_even = copy.deepcopy(lottery_ml)

lottery_ml_even['EVEN'] = 0

lottery_ml_even
def evenBalls(balls):

    numEven = 0

    for ball in balls:

        if(ball % 2 == 0):

            numEven += 1

    return numEven



indexsToDrop = []

currentBalls = []

for index, draw in lottery_ml_even.iterrows():

    currentBalls.append(draw['Ball 1'])

    currentBalls.append(draw['Ball 2'])

    currentBalls.append(draw['Ball 3'])

    currentBalls.append(draw['Ball 4'])

    currentBalls.append(draw['Ball 5'])

    currentBalls.append(draw['Ball 6'])

    numEven = evenBalls(currentBalls)

    if( numEven == 3):

        draw['EVEN'] = numEven

    else:

        draw['EVEN'] = numEven

        indexsToDrop.append(index)  

    currentBalls = []



lottery_ml_even = lottery_ml_even.drop(indexsToDrop)



del lottery_ml_even['EVEN']

lottery_ml_even
rows_to_retain_for_test = 350

number_of_rows= lottery_ml_even.values.shape[0] - rows_to_retain_for_test

games_window_size = 50 #amount of past games we need to take in consideration for training (It's also the number of draws)

number_of_features = lottery_ml_even.values.shape[1] #balls count
# Normalizing and downscaling the data such that mean per column is 0

scaler = StandardScaler().fit(lottery_ml_even.values)

transformed_dataset = scaler.transform(lottery_ml_even.values)

lottery_ml_normalized = pd.DataFrame(data=transformed_dataset, index=lottery_ml_even.index)

number_of_features = lottery_ml_even.values.shape[1] #balls count

lottery_ml_normalized
train, label, test = prepare_datasets(lottery_ml_normalized)

history = build_model(train, label, 500, 0.3)
# Plot training & validation accuracy values

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()