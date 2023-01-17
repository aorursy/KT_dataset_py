# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import warnings

warnings.filterwarnings("ignore")
#import data

data = pd.read_csv('/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv')
# Make a first look at the data

data.info()
data.head()

# This is significant for me to make a decision for making normalization.

# The values show me that a normalization process will be needed.
data["class"].unique()

# I have 2 types.
A = data[data["class"] == "Abnormal"]

N = data[data["class"] == "Normal"]
# int will be much useful for our algorithm. So let's change it.

data["class"] = [1 if each == 'Abnormal' else 0 for each in data["class"]]
data["class"].value_counts()
# I really don't know why but when I tried to visualize the data as 3D, I encountered with some problem that I can't find why these errors occur yet.

# Because of this problem I made a Scatter Plot and I added 3D plot on my final as an additional visualization.

# My 2D Scatter Plot includes the other features that I could not give place to my 3D plot.



plt.scatter(A.sacral_slope,A.degree_spondylolisthesis,color='r',label="Abnormal",alpha = 0.3)

plt.scatter(N.sacral_slope,N.degree_spondylolisthesis,color='g',label="Normal",alpha = 0.3)

plt.xlabel("sacral_slope")

plt.ylabel("degree_spondylolisthesis")

plt.legend()

plt.show()
y = data["class"].values

x_data = data.drop(["class"], axis=1)



# Normalization

x = (x_data- np.min(x_data)) / (np.max(x_data)-np.min(x_data)).values
#import train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5,random_state = 1)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3) # default neighbour = 5

knn.fit(x_train,y_train) # train the model

prediction = knn.predict(x_test) # with using this, you can see our machine's predictions
print("{} nn score: {}".format(3,knn.score(x_test,y_test)))
# k is hyperparameter, whixh means you have to adjust it, or:

# find k values

score_list = []

best_score = 0 # for score comparation

best_k = 0 # for finding best k value



for each in range(1,20):

    knn2 = KNeighborsClassifier(n_neighbors = each)

    knn2.fit(x_train,y_train) # modeli eğitiyoruz

    score_list.append(knn2.score(x_test,y_test))

    if (knn2.score(x_test,y_test) > best_score):

       best_score = knn2.score(x_test,y_test)

       best_k = each

    

plt.plot(range(1,20), score_list) # x_axis=range(1,100), yaxis=score_list

plt.xlabel("K values")

plt.ylabel("Accuracy")

plt.show()



print("Best score: ", best_score)

print("Best k values: ", best_k)

# Let's make a final comparation. I will create a dataframe which I can see predictions and test results much easier. 

prediction = knn.predict(x_test)

list_prediction = []

list_ytest = []

for each in range(0,len(prediction)):

    if prediction[each] == 1: 

       list_prediction.append("Abnormal")

    else: list_prediction.append("Normal")

print(list_prediction) # I wrote both of these to be sure my list is accurate,

print(prediction) # it means I want to be sure that 1's are for abnormal and 0's are for normal
list_ytest = []

for each in range(0,len(y_test)):

    if y_test[each] == 1: 

       list_ytest.append("Abnormal")

    else: list_ytest.append("Normal")

print(list_ytest) # I wrote both of these to be sure my list is accurate,

print(y_test) # it means I want to be sure that 1's are for abnormal and 0's are for normal
# Let's make a dataframe includes 2 of these 

accuracy = []

for each in range(0,30):

    if list_prediction[each:each+1] == list_ytest[each:each+1]:

        accuracy.append("Same")

    else: accuracy.append("Not Same")



list_label = ["Prediction","Test","Accuracy"]

list_col = [list_prediction[:30],list_ytest[:30],accuracy] # Check for first 30 values

zipped = list(zip(list_label,list_col)) # in order to create a new dataframe, zipped columns and labels

data_dict = dict(zipped) 

df_compare = pd.DataFrame(data_dict)

df_compare
# Now, let's make a quick look for how abnormality affects other features 

# I will make 3D Scatter Plot in order to show much more features

# Visualization

import plotly.graph_objects as go

import plotly.io as pio



trace1 = go.Scatter3d(

    x=A.pelvic_incidence,

    y=A.pelvic_radius,

    z=A.lumbar_lordosis_angle,

    name = "Abnormal",

    mode='markers',

    marker=dict(

        size=10,

        color='rgb(255,0,0)',   

    )

)



trace2 = go.Scatter3d(

    x=N.pelvic_incidence,

    y=N.pelvic_radius,

    z=N.lumbar_lordosis_angle,

    name = "Normal",

    mode='markers',

    marker=dict(

        size=10,

        color='rgb(0,0,255)',

    )

)



data = [trace1,trace2]

layout = go.Layout(

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0  

    ) 

)

fig = go.Figure(data=data, layout=layout)

pio.show(fig)
