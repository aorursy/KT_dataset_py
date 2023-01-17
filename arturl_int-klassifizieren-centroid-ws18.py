#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
%matplotlib inline
import matplotlib.pyplot as plt

#Als erstes lesen wir die .arff-Datei ein, die unsere Trainingsdaten enthält
def read_data(filename):
    f = open(filename)
    data_line = False
    data = []
    for l in f:
        l = l.strip() 
        if data_line:
            content = [float(x) for x in l.split(',')]
            if len(content) == 3:
                data.append(content)
        else:
            if l.startswith('@DATA'):
                data_line = True
    return data

train = read_data("../input/kiwhs-comp-1-complete/train.arff")
df_data = pd.DataFrame({'x':[item[0] for item in train], 'y':[item[1] for item in train], 'Category':[item[2] for item in train]})
df_data.head()
X = df_data[["x","y"]].values
Y = df_data["Category"].values
colors = {-1:'red',1:'blue'}

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,Y, random_state=0, test_size = 0.2)
def get_center(train, label, encoding):
    center_x = 0
    center_y = 0
    
    for i in range(len(train[:,0])):
        if label[i] == encoding:
            center_x += train[i,0]
    center_x = center_x/len(train[:,0])
    
    for i in range(len(train[:,1])):
        if label[i] == encoding:
            center_y += train[i,1]
    center_y = center_y/len(train[:,1])
    
    
    return (center_x, center_y)
center_red = get_center(X,Y,-1)
center_blue = get_center(X,Y,1)
print(center_red)
print(center_blue)

plt.figure(figsize=(5,5))
plt.scatter(X[:,0],X[:,1],c=df_data["Category"].apply(lambda x: colors[x]))
plt.scatter(center_red[0],center_red[1],c='green')
plt.scatter(center_blue[0],center_blue[1],c='yellow')
plt.xlabel("x")
plt.ylabel("y")
plt.show()
import math
def classify(x,y):
    distance_red = math.sqrt((center_red[0] - x)**2 + (center_red[1] - y)**2)
    distance_blue = math.sqrt((center_blue[0] - x)**2 + (center_blue[1] - y)**2)
    return -1 if distance_red < distance_blue  else 1
    
test_result = list()
for i in range(len(X_Test[:,0])):
    x = X_Test[i,0]
    y = X_Test[i,1]
    test_result.append(classify(x,y))
    
accuracy = 100 - 100.0 / len(Y_Test) * ((np.asarray(Y_Test) + np.asarray(test_result)) == 0).sum()   
print("Accuracy: ",accuracy)

Y_Color_result = list()
for encoding in test_result:
    Y_Color_result.append('red' if encoding == -1 else 'blue')   
    
plt.figure(figsize=(5,5))
plt.scatter(X_Test[:,0],X_Test[:,1],c=Y_Color_result)
plt.scatter(center_red[0],center_red[1],c='green')
plt.scatter(center_blue[0],center_blue[1],c='yellow')
plt.xlabel("x")
plt.ylabel("y")
plt.show()
######### hier versuchen wir nun das Vorhersagen#######
testdf = pd.read_csv("../input/kiwhs-comp-1-complete/test.csv")

testX = testdf[["X","Y"]].values
######################################################

test_result = list()
for i in range(len(testX[:,0])):
    x = testX[i,0]
    y = testX[i,1]
    test_result.append(classify(x,y))

######## Anschließend Speichern wir unsere Vorhersage ab #######
prediction = pd.DataFrame()
id = []
for i in range(len(testX)):
    id.append(i)
    i = i + 1
prediction["Id (String)"] = id 
prediction["Category (String)"] = test_result
print(prediction[:10])
prediction.to_csv("predict.csv", index=False)
##################### ENDE ####################################
Y_Color_result = list()
for encoding in test_result:
    Y_Color_result.append('red' if encoding == -1 else 'blue')   

plt.figure(figsize=(5,5))
plt.scatter(testX[:,0],testX[:,1],c=Y_Color_result)
plt.scatter(center_red[0],center_red[1],c='green')
plt.scatter(center_blue[0],center_blue[1],c='yellow')
plt.xlabel("x")
plt.ylabel("y")
plt.show()
#Plotting test-data classification and boundary
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
X = testX
y = Y_Color_result 
h = .02  # step size in the mesh
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 0.25, X[:, 0].max() + 0.25
y_min, y_max = X[:, 1].min() - 0.25, X[:, 1].max() + 0.25
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

points = np.c_[xx.ravel(),yy.ravel()]

Z = np.zeros(len(points))
for i in range(len(points)):
    temp_x = points[i][0]
    temp_y = points[i][1]
    Z[i] = classify(temp_x,temp_y)
    

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(figsize=(5,5))
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=10)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.show()