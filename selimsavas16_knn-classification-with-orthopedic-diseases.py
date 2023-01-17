import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv("../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")
data.head() #pelvic_tilt numeric isimlendirmesi bizim için sıkıntı  bir isimlendirme
data_columns_v2 = ["pelvic_incidence","pelvic_tilt_numeric","lumbar_lordosis_angle","sacral_slope","pelvic_radius","degree_spondylolisthesis","class"]
data.columns = data_columns_v2
data.head()
data.info()
# Let's int the properties with string values

data["class"] = [1 if each == "Abnormal" else 0 for each in data["class"]]
data.head()
# For train and test : Determining the x and y values.

y = data["class"].values

x_data = data.drop(["class"], axis=1)
# Normaization

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))
# Let's set the data according to the class values

ANO = data[data["class"] == 1]

NO = data[data["class"] == 0]
ANON = ANO["class"].count()
NON = NO["class"].count()

labels = "Abnormal","Normal"
sizes = [ANON,NON]
explode = (0, 0.3)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode,labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title("Abnormal and Normal")

plt.show()
# Let's set the data according to the class values

ANO = data[data["class"] == 1]

NO = data[data["class"] == 0]
import plotly.graph_objects as go

x_data = [["Abnormal","Normal"],["Abnormal","Normal"],["Abnormal","Normal"],["Abnormal","Normal"],["Abnormal","Normal"],["Abnormal","Normal"]]

y0 = ANO["pelvic_incidence"]
y1 = NO["pelvic_incidence"]

y2 = ANO["pelvic_tilt_numeric"]
y3 = NO["pelvic_tilt_numeric"]

y4 = ANO["lumbar_lordosis_angle"]
y5 = NO["lumbar_lordosis_angle"]

y6 = ANO["sacral_slope"]
y7 = NO["sacral_slope"]

y8 = ANO["pelvic_radius"]
y9 = NO["pelvic_radius"]

y10 = ANO["degree_spondylolisthesis"]
y11 = NO["degree_spondylolisthesis"]


y_data = [[y0, y1],[y2, y3],[y4,y5],[y6,y7],[y8,y9],[y10,y11]]

colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)',
          'rgba(255, 65, 54, 0.5)', 'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)','rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)',
          'rgba(255, 65, 54, 0.5)', 'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)']

fig = plt.figure(figsize=(6,2))

for xd, yd, cls,columns in zip(x_data, y_data, colors,data.columns):
        
    fig= go.Figure()
    fig.add_trace(go.Box(
        y=yd[0],
        name=xd[0],
        boxpoints='all',
        jitter=0.5,
        whiskerwidth=0.2,
        fillcolor=cls,
        marker_size=2,
        line_width=1))
        
    fig.add_trace(go.Box(
        y=yd[1],
        name=xd[1],
        boxpoints='all',
        jitter=0.5,
        whiskerwidth=0.2,
        fillcolor=cls,
        marker_size=2,
        line_width=1),)
        

    fig.update_layout(
        yaxis=dict(
                                                                                            
            title=columns,
            autorange=True,
            showgrid=True,
            zeroline=True,
            dtick=5,
            gridcolor='rgb(255, 255, 255)',
            gridwidth=1,
            zerolinecolor='rgb(255, 255, 255)',
            zerolinewidth=2,),
        margin=dict(
            l=40,
            r=30,
            b=40,
            t=100,
        ),
        paper_bgcolor='rgb(243, 243, 243)',
        plot_bgcolor='rgb(243, 243, 243)',
        showlegend=False
    )

    fig.show()
    
   
plt.show()
plt.figure(figsize=(15,10))

plt.subplot(231)
sns.boxplot(data["class"],data["pelvic_incidence"])
plt.subplot(232)
sns.boxplot(data["class"],data["pelvic_tilt_numeric"])
plt.subplot(233)
sns.boxplot(data["class"],data["lumbar_lordosis_angle"])
plt.subplot(234)
sns.boxplot(data["class"],data["sacral_slope"])
plt.subplot(235)
sns.boxplot(data["class"],data["pelvic_radius"])
plt.subplot(236)
sns.boxplot(data["class"],data["degree_spondylolisthesis"])


plt.show()
data.corr()
sns.pairplot(data, hue="class")
# Train Test Split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=43)
# knn model 2

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3) # k

knn.fit(x_train,y_train)
# predict 1

predict = knn.predict(x_test)
predict
print(" {} nn score: {} ".format(3,knn.score(x_test,y_test)))
# find k value
score_list = []
for each in range(1,15):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
    
plt.figure(figsize=(20,10))
plt.plot(range(1,15),score_list)

plt.show()
# knn model 2

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 11) # k

knn.fit(x_train,y_train)
# predict 2

predict = knn.predict(x_test)
predict
print(" {} nn score: {} ".format(11,knn.score(x_test,y_test)))
predict
y_test
Test = np.concatenate((y_test,predict), axis=0)

Test = Test.reshape(2,93)

Test
x_test
deger = x_test
deger = deger.reset_index() 
deger["Real_Values"] = ["Abnormal" if each == 0 else "Normal" for each in y_test]
deger["Test_Values"] = ["Abnormal" if each == 0 else "Normal" for each in predict]
deger["Difference"] = -(y_test - predict)

deger
