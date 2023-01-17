# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



print('test')
class ETL(object):



    """

    Owner: Jens Ponnet

    Date: 15/07/2016

    Functionality: This class offers methods for extracting, transforming and loading data

    Included methods:

        - extractFromCSV: extracts data from a CSV file and returns it into a pandas data frame

    """

    import numpy as np

    import pandas as pd

    import matplotlib.pyplot as plt

    from sklearn import preprocessing

    

    def extractFromCSV(self, filepath):

        if type(filepath) == str:

            df = pd.read_csv(filepath, delimiter=",")

            return df

        else:

            return "filepath is not a string"



    def splitXY(self,inputselection,outputselection,filepath):

        df = self.extractFromCSV(filepath)

        list1 = np.array(inputselection)

        list2 = np.array(outputselection)

        x = df.as_matrix(columns=list1.tolist())

        y = df.as_matrix(columns=list2.tolist())

        # Y = np.reshape(np.array(df[outputselection]),(-1,1))

        return x, y



    def normalize(self,dataset):

        min_max_scaler = preprocessing.MinMaxScaler()

        dataset = min_max_scaler.fit_transform(dataset)

        return dataset



    def scaleback(self,dataset):

        min_max_scaler = preprocessing.MinMaxScaler()

        dataset = min_max_scaler.inverse_transform(dataset)

        return dataset



    def takeOutValidationData(self, dataset, number):

            total_rows = len(dataset)

            dataset_holdin = dataset[:(total_rows-1-number)]

            dataset_holdout = dataset[(total_rows-number):]

            return dataset_holdin, dataset_holdout
'''

generate scatter on leavers versus stayers

'''



import matplotlib.pyplot as plt

import pandas

import numpy as np

from sklearn import preprocessing

from scipy.stats import itemfreq

from mpl_toolkits.mplot3d import Axes3D

from sklearn import svm

from sklearn.model_selection import train_test_split

from sklearn import preprocessing





def generate_scatter(x, y, subplot_rows, subplot_columns, points=None, colormap=None, legend=False):

    

    """

    Owner: Jens Ponnet

    Date: 15/11/2016

    Functionality: This function offers a 4x4 scatterplot through a easy to use interface parameters

    """

    figure = plt.figure(figsize=(9, 7), dpi=100)

    grapharray = []

    n_graphs = subplot_rows * subplot_columns



    for i in range(1,n_graphs+1):

        graph = figure.add_subplot(subplot_rows, subplot_columns, i)

        grapharray.append(graph)



    for i in range(0,n_graphs):

        grapharray[i].scatter(np.array(x[:,i]), np.array(y[:,i]), c=points , cmap=colormap)

        grapharray[i].set_xlabel(x_axis[i])

        grapharray[i].set_ylabel(y_axis[i])

        if legend == True:

            grapharray[i].legend(np.unique(points))

        else:

            continue



    plt.show()



pp = ETL()

le = preprocessing.LabelEncoder()

le.fit(['low','medium','high'])





x_axis = ['satisfaction_level', 'satisfaction_level', 'average_montly_hours', 'satisfaction_level']

y_axis = ['last_evaluation', 'average_montly_hours', 'last_evaluation', 'time_spend_company']

x,y = pp.splitXY(x_axis, y_axis, "../input/HR_comma_sep.csv")

colors= pp.extractFromCSV("../input/HR_comma_sep.csv")

colors = np.array(colors['left'])

generate_scatter(x, y, 2, 2, colors, 'bwr')
'''

generate scatter on leavers with a breakdown per salary

'''



x_axis2 = ['satisfaction_level', 'average_montly_hours']

y_axis2 = ['last_evaluation', 'last_evaluation']

x2,y2 = pp.splitXY(x_axis2, y_axis2, "../input/HR_comma_sep.csv")

colors1 = pp.extractFromCSV("../input/HR_comma_sep.csv")

colors2 = np.array(colors1['left'])

colors3 = np.array(colors1['salary'])



test1c = [i1 for i11,i1 in enumerate(colors3) if colors2[i11] == 1]

test1x = [i1 for i11,i1 in enumerate(x2[:,1]) if colors2[i11] == 1]

test1y = [i1 for i11,i1 in enumerate(y2[:,1]) if colors2[i11] == 1]

test1xl = [i1 for i11,i1 in enumerate(test1x) if test1c[i11] == 'low']

test1yl = [i1 for i11,i1 in enumerate(test1y) if test1c[i11] == 'low']

test1cl = [item for item in test1c if item == 'low']

test1xm = [i1 for i11,i1 in enumerate(test1x) if test1c[i11] == 'medium']

test1ym = [i1 for i11,i1 in enumerate(test1y) if test1c[i11] == 'medium']

test1cm = [item for item in test1c if item == 'medium']

test1xh = [i1 for i11,i1 in enumerate(test1x) if test1c[i11] == 'high']

test1yh = [i1 for i11,i1 in enumerate(test1y) if test1c[i11] == 'high']

test1ch = [item for item in test1c if item == 'high']

test1cl = le.transform(test1cl)

test1cm = le.transform(test1cm)

test1ch = le.transform(test1ch)



figure = plt.figure(figsize=(9, 7), dpi=100)

graph = figure.add_subplot(1, 1, 1)

graph.scatter(test1xl, test1yl, c='red', label = 'low')

graph.scatter(test1xm, test1ym, c='blue', label = 'medium')

graph.scatter(test1xh, test1yh, c='yellow', label = 'high')

graph.legend()

graph.set_xlabel(x_axis2[1])

graph.set_ylabel(y_axis2[1])



plt.show()
'''

generate scatter on leavers with a breakdown per salary

'''



x_axis2 = ['average_montly_hours', 'satisfaction_level']

y_axis2 = ['last_evaluation', 'last_evaluation']

x2,y2 = pp.splitXY(x_axis2, y_axis2, "../input/HR_comma_sep.csv")

colors1 = pp.extractFromCSV("../input/HR_comma_sep.csv")

colors2 = np.array(colors1['left'])

colors3 = np.array(colors1['salary'])



test1c = [i1 for i11,i1 in enumerate(colors3) if colors2[i11] == 1]

test1x = [i1 for i11,i1 in enumerate(x2[:,1]) if colors2[i11] == 1]

test1y = [i1 for i11,i1 in enumerate(y2[:,1]) if colors2[i11] == 1]

test1xl = [i1 for i11,i1 in enumerate(test1x) if test1c[i11] == 'low']

test1yl = [i1 for i11,i1 in enumerate(test1y) if test1c[i11] == 'low']

test1cl = [item for item in test1c if item == 'low']

test1xm = [i1 for i11,i1 in enumerate(test1x) if test1c[i11] == 'medium']

test1ym = [i1 for i11,i1 in enumerate(test1y) if test1c[i11] == 'medium']

test1cm = [item for item in test1c if item == 'medium']

test1xh = [i1 for i11,i1 in enumerate(test1x) if test1c[i11] == 'high']

test1yh = [i1 for i11,i1 in enumerate(test1y) if test1c[i11] == 'high']

test1ch = [item for item in test1c if item == 'high']

test1cl = le.transform(test1cl)

test1cm = le.transform(test1cm)

test1ch = le.transform(test1ch)



figure = plt.figure(figsize=(9, 7), dpi=100)

graph = figure.add_subplot(1, 1, 1)

graph.scatter(test1xl, test1yl, c='red', label = 'low')

graph.scatter(test1xm, test1ym, c='blue', label = 'medium')

graph.scatter(test1xh, test1yh, c='yellow', label = 'high')

graph.legend()

graph.set_xlabel(x_axis2[1])

graph.set_ylabel(y_axis2[1])



plt.show()
'''

generate bar charts on time work accident

'''

x_axis = ['Work_accident']

y_axis = ['left']

x,y = pp.splitXY(x_axis, y_axis, "../input/HR_comma_sep.csv")

figure = plt.figure(figsize=(9, 7), dpi=100)

graph = figure.add_subplot(1, 1, 1)

y2 = np.array([i1 for i11,i1 in enumerate(x) if y[i11] == 1])

y2 = itemfreq(y2)

y2 = y2[:,1]

# y2 = np.append(y2, [0,0,0])

y3 = np.array([i1 for i11,i1 in enumerate(x) if y[i11] == 0])

y3 = itemfreq(y3)

y3 = y3[:,1]

x = itemfreq(x)

x1 = x[:,0]

y1 = x[:,1]



pos = list(range(len(x1)))

width = 0.25

graph.bar(pos, y1, width, color='#EE3224', label='Total', alpha=0.5)

graph.bar([p + width for p in pos], y2, width, color='#F78F1E', label='Left', alpha=0.5)

graph.bar([p + width*2 for p in pos], y3, width,color='#FFC222', label='Stayed', alpha=0.5)

graph.legend(['Total', 'Left', 'Stayed'], loc='upper right')

graph.set_ylabel('# of employees')

graph.set_xticks([p + 1.5 * width for p in pos])

graph.set_xticklabels(x[:,0])

# graph.set_title('Number of years at the company')

# graph.set_title('Work accident (0=no,1=yes)')

# graph.set_title('Promotion last 5 years (0=no,1=yes)')

graph.set_title('Work accident')

plt.xlim(min(pos)-width, max(pos)+width*4)

plt.grid()

plt.show()
x_axis = ['sales']

y_axis = ['left']

x,y = pp.splitXY(x_axis, y_axis, "../input/HR_comma_sep.csv")

figure = plt.figure(figsize=(9, 7), dpi=100)

graph = figure.add_subplot(1, 1, 1)

y2 = np.array([i1 for i11,i1 in enumerate(x) if y[i11] == 1])

y2 = itemfreq(y2)

y2 = y2[:,1]

# y2 = np.append(y2, [0,0,0])

y3 = np.array([i1 for i11,i1 in enumerate(x) if y[i11] == 0])

y3 = itemfreq(y3)

y3 = y3[:,1]

x = itemfreq(x)

x1 = x[:,0]

y1 = x[:,1]



pos = list(range(len(x1)))

width = 0.25

graph.bar(pos, y1, width, color='#EE3224', label='Total', alpha=0.5)

graph.bar([p + width for p in pos], y2, width, color='#F78F1E', label='Left', alpha=0.5)

graph.bar([p + width*2 for p in pos], y3, width,color='#FFC222', label='Stayed', alpha=0.5)

graph.legend(['Total', 'Left', 'Stayed'], loc='upper right')

graph.set_ylabel('# of employees')

graph.set_xticks([p + 1.5 * width for p in pos])

graph.set_xticklabels(x[:,0])

# graph.set_title('Number of years at the company')

# graph.set_title('Work accident (0=no,1=yes)')

# graph.set_title('Promotion last 5 years (0=no,1=yes)')

graph.set_title('Department')

plt.xlim(min(pos)-width, max(pos)+width*4)

plt.grid()

plt.show()
x_axis = ['promotion_last_5years']

y_axis = ['left']

x,y = pp.splitXY(x_axis, y_axis, "../input/HR_comma_sep.csv")

figure = plt.figure(figsize=(9, 10), dpi=100)

graph = figure.add_subplot(1, 1, 1)

y2 = np.array([i1 for i11,i1 in enumerate(x) if y[i11] == 1])

y2 = itemfreq(y2)

y2 = y2[:,1]

# y2 = np.append(y2, [0,0,0])

y3 = np.array([i1 for i11,i1 in enumerate(x) if y[i11] == 0])

y3 = itemfreq(y3)

y3 = y3[:,1]

x = itemfreq(x)

x1 = x[:,0]

y1 = x[:,1]



pos = list(range(len(x1)))

width = 0.25

graph.bar(pos, y1, width, color='#EE3224', label='Total', alpha=0.5)

graph.bar([p + width for p in pos], y2, width, color='#F78F1E', label='Left', alpha=0.5)

graph.bar([p + width*2 for p in pos], y3, width,color='#FFC222', label='Stayed', alpha=0.5)

graph.legend(['Total', 'Left', 'Stayed'], loc='upper right')

graph.set_ylabel('# of employees')

graph.set_xticks([p + 1.5 * width for p in pos])

graph.set_xticklabels(x[:,0])

# graph.set_title('Number of years at the company')

# graph.set_title('Work accident (0=no,1=yes)')

# graph.set_title('Promotion last 5 years (0=no,1=yes)')

graph.set_title('promotion_last_5years')

plt.xlim(min(pos)-width, max(pos)+width*4)

plt.grid()

plt.show()
x_axis = ['time_spend_company']

y_axis = ['left']

x,y = pp.splitXY(x_axis, y_axis, "../input/HR_comma_sep.csv")

figure = plt.figure(figsize=(9, 10), dpi=100)

graph = figure.add_subplot(1, 1, 1)

y2 = np.array([i1 for i11,i1 in enumerate(x) if y[i11] == 1])

print(y2)

y2 = itemfreq(y2)

print(y2)

y2 = y2[:,1]

print(y2)

y2 = np.append(y2, [0,0,0])

y3 = np.array([i1 for i11,i1 in enumerate(x) if y[i11] == 0])

y3 = itemfreq(y3)

y3 = y3[:,1]

x = itemfreq(x)

x1 = x[:,0]

y1 = x[:,1]



pos = list(range(len(x1)))

width = 0.25

graph.bar(pos, y1, width, color='#EE3224', label='Total', alpha=0.5)

graph.bar([p + width for p in pos], y2, width, color='#F78F1E', label='Left', alpha=0.5)

graph.bar([p + width*2 for p in pos], y3, width,color='#FFC222', label='Stayed', alpha=0.5)

graph.legend(['Total', 'Left', 'Stayed'], loc='upper right')

graph.set_ylabel('# of employees')

graph.set_xticks([p + 1.5 * width for p in pos])

graph.set_xticklabels(x[:,0])

# graph.set_title('Number of years at the company')

# graph.set_title('Work accident (0=no,1=yes)')

# graph.set_title('Promotion last 5 years (0=no,1=yes)')

graph.set_title('promotion_last_5years')

plt.xlim(min(pos)-width, max(pos)+width*4)

plt.grid()

plt.show()
'''

generate 3D plot for 'last_evaluation', 'average_montly_hours', 'satisfaction_level'

'''

x_axis = ['satisfaction_level']

y_axis = ['last_evaluation']

z_axis = ['average_montly_hours']

zz_axis = ['left']

x,y = pp.splitXY(x_axis, y_axis, "../input/HR_comma_sep.csv")

z,zz = pp.splitXY(z_axis, zz_axis, "../input/HR_comma_sep.csv")

x_leavers = [i1 for i11,i1 in enumerate(x) if zz[i11] == 1]

y_leavers = [i1 for i11,i1 in enumerate(y) if zz[i11] == 1]

z_leavers = [i1 for i11,i1 in enumerate(z) if zz[i11] == 1]



fig = plt.figure(figsize=(9, 7), dpi=100)

ThreeDplot = fig.add_subplot(111, projection='3d')

ThreeDplot.scatter(x_leavers,y_leavers,z_leavers)

ThreeDplot.set_xlabel('satisfaction level')

ThreeDplot.set_ylabel('last evaluation')

ThreeDplot.set_zlabel('average monthly hours')

plt.show()
'''

Using support vector machine on 2 variables + output of a 2d graph to present the results

'''

x_axis = ['average_montly_hours', 'last_evaluation']

y_axis = ['left']

x, y = pp.splitXY(x_axis, y_axis, "../input/HR_comma_sep.csv")

y = y.ravel()



# we create an instance of SVM and fit out data. We do not scale our

# data since we want to plot the support vectors





def svm_2d_graph(x, y, x_label='', y_label='', title='SVC', kernel='rbf', gamma=10):

    # setup & train the support vector machine

    # C = 1  # SVM regularization parameter

    clf = svm.SVC(probability=True, kernel=kernel, gamma=gamma)

    clf.fit(x, y)

    # create a meshgrid

    h = .02  # step size in the mesh

    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1

    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),

                         np.arange(y_min, y_max, h))

    # create the contour predictions

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    # Put the result of the predictions into a color plot

    plt.figure(figsize=(9, 7), dpi=100)

    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot the training points

    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.coolwarm)

    # put labels in

    plt.xlabel(x_label)

    plt.ylabel(y_label)

    plt.xlim(80, 350)

    plt.ylim(0.35, 1.05)

    plt.xticks()

    plt.yticks()

    plt.title(title)

    plt.show()



svm_2d_graph(x, y, 'average_montly_hours', 'last_evaluation', 'test', 'rbf', 10)
x_axis = ['average_montly_hours', 'last_evaluation', 'satisfaction_level', 'time_spend_company']

y_axis = ['left']

x, y = pp.splitXY(x_axis, y_axis, "../input/HR_comma_sep.csv")

# x[:,4] = labelencoder.fit_transform(x[:,4])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

y = y.ravel()



clf = svm.SVC(probability=True, kernel='rbf', gamma=10)

clf.fit(x, y)



# creating an instance of SVM and fit out data. We do not scale our

# data since we want to plot the support vectors

mean_acc = clf.score(x_test, y_test)

print(mean_acc)