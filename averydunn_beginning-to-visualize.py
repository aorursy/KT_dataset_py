# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 
import matplotlib.axes
from wordcloud import WordCloud, STOPWORDS

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# plotting random points 

random = np.random.rand(10)
df = pd.DataFrame(random, columns=['random'])
df
# Now we can plot the random points above just by calling the plot function as 
# shown below 
df.plot()
df.plot(kind='area')
df.plot(kind='hist')
df.plot(kind='box')
df.plot(kind='kde')
# We can orient the random points in a scatter plot as well
# plt.scatter() takes an x and y argument, whereas the above did not
# We are just randomly plotting the points on the y axis across the x axis 
# so the x points are just the index

random = np.random.rand(50)

df = pd.DataFrame(random, columns=['random'])
plt.scatter(df.index, df)
# gather data for the pie chart 

# here we can use the WashU breakdown of schools and number of students 

architecture = 183
business = 836
art = 304
artsci = 4490
engineering = 1356
other = 622

schools = [architecture, business, art, artsci, engineering, other]
labels = 'architecture', 'business', 'art', 'artsci', 'engineering', 'other'
plt.pie(schools,labels=labels,autopct='%1.1f%%')
plt.title('Undergraduate Enrollment by School')
plt.axis('equal')
plt.show()

schools_df = pd.DataFrame(schools, columns=['School'])

colors = ['salmon', 'pink', 'lightblue', 'green', 'yellow']
explode = (0, 0, 0, 0.1, 0, 0)

plt.pie(schools_df, labels=labels, autopct='%1.1f%%', 
        startangle=15, shadow = True, colors=colors, explode=explode)
plt.title('Undergraduate Enrollment by School')
plt.axis('equal')
plt.show()

plt.bar(labels, schools, color=['red', 'black', 'pink', 'orange', 'green', 'blue'])

plt.barh(labels, schools)
# creating a dataframe of sentence 
data = {'sentences': ['WashU', 'McKelvey School of Engineering', 
                     'Computer Engineering', 'St.Louis', 'semester', 
                     'Fall', 'Math', 'Sciences', 'Computer Science', 'Data Science' 
                     'WashU']}
words_df = pd.DataFrame(data, columns=['sentences'])
words_df


def createWordCloud(df): 
    # initializing what will be our big string of words 
    comment_words = ''
    # setting stopwords which WordCloud conveniently has that are 'unnecessary'
    stopwords = set(STOPWORDS)
    for i in df: 
        i = str(i)
        tokens = i.split()
#         for j in range(len(tokens)): 
#             tokens[j] = tokens[j].lower()
        comment_words += ' '.join(tokens) + ' '
    wordcloud = WordCloud(width=800, height=800,
                        background_color = 'white', stopwords=stopwords, 
                         min_font_size = 10).generate(comment_words)
    # plot wordcloud image 
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()

createWordCloud(words_df.sentences)
cereals = pd.read_csv('/kaggle/input/80-cereals/cereal.csv')
cereals
# scatter plot with line of best fit 
m,b = np.polyfit(cereals.calories, cereals.sugars, 1)

plt.scatter(cereals.calories, cereals.sugars)
plt.plot(cereals.calories, m*cereals.calories + b)
plt.xlabel('Calories')
plt.ylabel('Sugar')
plt.title('Are Calories and Sugar Related?')

plt.show()
m,b = np.polyfit(cereals.calories, cereals.sodium, 1)

plt.scatter(cereals.calories, cereals.sodium)
plt.plot(cereals.calories, m*cereals.calories+b)
plt.xlabel('Calories')
plt.ylabel('Sodium')
plt.title('Are sodium and calories related?')

plt.show()
m,b = np.polyfit(cereals.calories, cereals.protein, 1)

plt.scatter(cereals.calories, cereals.protein)
plt.plot(cereals.calories, m*cereals.calories+b)
plt.xlabel('Calories')
plt.ylabel('Protein')
plt.title('Are calories and protein related?')

plt.show()
m,b = np.polyfit(cereals.calories, cereals.carbo, 1)

plt.scatter(cereals.calories, cereals.carbo)
plt.plot(cereals.calories, m*cereals.calories+b)
plt.xlabel('Calories')
plt.ylabel('Carbs')
plt.title('Are calories and carbs related?')

plt.show()
m,b = np.polyfit(cereals.calories, cereals.fat, 1)

plt.scatter(cereals.calories, cereals.fat)
plt.plot(cereals.calories, m*cereals.calories+b)
plt.xlabel('Calories')
plt.ylabel('Fat')
plt.title('Are calories and fat related?')

plt.show()
m,b = np.polyfit(cereals.sugars, cereals.rating, 1)
plt.scatter(cereals.sugars, cereals.rating)
plt.plot(cereals.sugars, m*cereals.sugars+b)

plt.xlabel('Sugar')
plt.ylabel('Rating')
plt.title('Does more sugar mean higher rating?')
plt.show()
plt.scatter(cereals.rating, cereals.shelf)
plt.xlabel('Rating')
plt.ylabel('Shelf')
plt.title('Does higher shelf lead to better rating?')
plt.show()
cereals[cereals['rating'] > 90]
head = cereals.head(4)
head
fig = plt.figure(figsize=(7,7))

list_percents = []
for carbs, fat, protein in zip(head.carbo, head.fat, head.protein):
    total = float(carbs + fat + protein)
    percent_carbs = float(carbs/total)*100
    percent_fat = float(fat/total)*100
    percent_protein = float(protein/total)*100
    labels = 'carbs', 'fat', 'protein'
    total_percent = [percent_carbs, percent_fat, percent_protein]
    list_percents.append(total_percent)
for i in range(1, len(head)+1): 
    ax = fig.add_subplot(2, 2, i)
    ax.pie(list_percents[i-1], labels=labels, autopct='%1.1f%%')
    ax.set_title(head.name[i-1])
plt.tight_layout()
plt.show()
        
        
    
    
        
        
    
    
    # creating pie charts
    
    
createWordCloud(cereals.name)
calCup = []
for cal, cup in zip(cereals.calories, cereals.cups): 
    cal_per_cup = round(float(cal/cup),2)
    calCup.append(cal_per_cup)
cereals['calories_per_cup'] = calCup
cereals
cereals[cereals['calories_per_cup'] == cereals['calories_per_cup'].max()]
plt.boxplot(cereals.calories_per_cup)
# Identifying outliers 
cereals[cereals.calories_per_cup > 225]
boxplot = cereals.boxplot(column=['calories_per_cup'], flierprops=dict(markerfacecolor='r', marker='s', label='not shown'))
boxplot.annotate('Mueslix Crispy Blend', (1, 238.81), xytext=(0.6, 0.3),
    textcoords='axes fraction',
    arrowprops=dict(facecolor='black', arrowstyle='wedge'),
    fontsize=11)
boxplot.annotate('Oatmeal Raisin Crisp', (1, 260.00), xytext=(0.6, 0.45), textcoords='axes fraction', 
                arrowprops=dict(facecolor='black', arrowstyle='wedge'), fontsize=11)
boxplot.annotate('Great Grains Pecan', (1, 363.64), xytext=(0.6, 0.65), textcoords='axes fraction', 
                arrowprops=dict(facecolor='black', arrowstyle='wedge'), fontsize=11)
boxplot.annotate('Grape-Nuts', (1, 440.00), xytext=(0.6, 0.85), textcoords='axes fraction', 
                arrowprops=dict(facecolor='black', arrowstyle='wedge'), fontsize=11)
plt.show()
cereals.groupby('mfr').mfr.count()
A_list = []
G_list = []
K_list = []
N_list = []
P_list = []
Q_list = []
R_list = []
for m, cal in zip(cereals.mfr, cereals.calories_per_cup): 
    if m=='A': 
        A_list.append(cal)
    elif m=='G': 
        G_list.append(cal)
    elif m=='K': 
        K_list.append(cal)
    elif m=='N': 
        N_list.append(cal)
    elif m=='P': 
        P_list.append(cal)
    elif m=='Q': 
        Q_list.append(cal)
    else: 
        R_list.append(cal)
print('Ralston Purina Avg Calories per Cup: ', round(sum(R_list)/len(R_list),2))
print('Quaker Oats Avg Calories per Cup: ', round(sum(Q_list)/len(Q_list), 2))
print('Post Avg Calories per Cup: ', round(sum(P_list)/len(P_list), 2))
print('Nabisco Avg Calories per Cup: ', round(sum(N_list)/len(N_list), 2))
print('Kellogg Avg Calories per Cup: ', round(sum(K_list)/len(K_list), 2))
print('General Mills Avg Calories per Cup: ', round(sum(G_list)/len(G_list), 2))
print('American Home Food Producsts Avg Calories per Cup: ', round(sum(A_list)/len(A_list), 2))
    
cal_R = round(sum(R_list)/len(R_list), 2)
cal_Q = round(sum(Q_list)/len(Q_list), 2)
cal_P = round(sum(P_list)/len(P_list), 2)
cal_N = round(sum(N_list)/len(N_list), 2)
cal_K = round(sum(K_list)/len(K_list), 2)
cal_G = round(sum(G_list)/len(G_list), 2)
cal_A = round(sum(A_list)/len(A_list), 2)

data = [R_list, Q_list, P_list, N_list, K_list, G_list, A_list]

list_cals = [cal_R, cal_Q, cal_P, cal_N, cal_K, cal_G, cal_A]
labels = ['Ralston Purina', 'Quaker Oats', 'Post', 'Nabisco', 'Kellogg', 'General Mills', 'American Home Food Producsts']
# What is the best way to visualize these values? 
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
ax1.barh(labels, list_cals)
ax2.pie(list_cals, labels=labels, autopct='%1.1f%%')
ax3.boxplot(data, labels=labels)
plt.xticks(rotation=45)