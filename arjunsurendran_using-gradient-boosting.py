#Importing Libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('ggplot')

import os
#Importing Dataset

sudoku = pd.read_csv('../input/sudoku.csv')
#Preview of Data

sudoku.head()
quizzes = np.zeros((1000000, 81), np.int32)

solutions = np.zeros((1000000, 81), np.int32)

for i, line in enumerate(open('../input/sudoku.csv', 'r').read().splitlines()[1:]):

    quiz, solution = line.split(",")

    for j, q_s in enumerate(zip(quiz, solution)):

        q, s = q_s

        quizzes[i, j] = q

        solutions[i, j] = s

quizzes = quizzes.reshape((-1, 9, 9))

solutions = solutions.reshape((-1, 9, 9))
#Sample Puzzle

print(quizzes[0])
#Sample Solution

print(solutions[0])
X = []

y = []
for p in range(1000):

    k = quizzes[p]

    m = solutions[p]

    for i in range(9):

        for j in range(9):

            temp = []

            value = m[i][j]

            for l in range(i):

                temp.append(k[l][j])

            for l in range(i+1,9):

                temp.append(k[l][j])

            for l in range(j):

                temp.append(k[i][l])

            for l in range(j+1,9):

                temp.append(k[i][l])

            temp.append(i)

            temp.append(j)

            

            if i<3 and j<3:

                for l in range(0,3):

                    for r in range(0,3):

                        if l!= i:

                            temp.append(k[l][r])

                        else:

                            if r!=j:

                                temp.append(k[l][r])

                X.append(temp)

                y.append(value)

            elif i<3 and j<6:

                for l in range(0,3):

                    for r in range(3,6):

                        if l!= i:

                            temp.append(k[l][r])

                        else:

                            if r!=j:

                                temp.append(k[l][r])

                X.append(temp)

                y.append(value)

            elif i<3 and j<9:

                for l in range(0,3):

                    for r in range(6,9):

                        if l!= i:

                            temp.append(k[l][r])

                        else:

                            if r!=j:

                                temp.append(k[l][r])

                X.append(temp)

                y.append(value)

                

            elif i<6 and j<3:

                for l in range(3,6):

                    for r in range(0,3):

                        if l!= i:

                            temp.append(k[l][r])

                        else:

                            if r!=j:

                                temp.append(k[l][r])

                X.append(temp)

                y.append(value)

            elif i<6 and j<6:

                for l in range(3,6):

                    for r in range(3,6):

                        if l!= i:

                            temp.append(k[l][r])

                        else:

                            if r!=j:

                                temp.append(k[l][r])

                X.append(temp)

                y.append(value)

            elif i<6 and j<9:

                for l in range(3,6):

                    for r in range(6,9):

                        if l!= i:

                            temp.append(k[l][r])

                        else:

                            if r!=j:

                                temp.append(k[l][r])

                X.append(temp)

                y.append(value)

                

            elif i<9 and j<3:

                for l in range(6,9):

                    for r in range(0,3):

                        if l!= i:

                            temp.append(k[l][r])

                        else:

                            if r!=j:

                                temp.append(k[l][r])

                X.append(temp)

                y.append(value)

            elif i<9 and j<6:

                for l in range(6,9):

                    for r in range(3,6):

                        if l!= i:

                            temp.append(k[l][r])

                        else:

                            if r!=j:

                                temp.append(k[l][r])

                X.append(temp)

                y.append(value)

            elif i<9 and j<9:

                for l in range(6,9):

                    for r in range(6,9):

                        if l!= i:

                            temp.append(k[l][r])

                        else:

                            if r!=j:

                                temp.append(k[l][r])

                X.append(temp)

                y.append(value)

                

            

        for j in range(9):

            temp = []

            value = m[i][j]

            for l in range(i):

                temp.append(m[l][j])

            for l in range(i+1,9):

                temp.append(m[l][j])

            for l in range(j):

                temp.append(m[i][l])

            for l in range(j+1,9):

                temp.append(m[i][l])

            temp.append(i)

            temp.append(j)

            

            if i<3 and j<3:

                for l in range(0,3):

                    for r in range(0,3):

                        if l!= i:

                            temp.append(m[l][r])

                        else:

                            if r!=j:

                                temp.append(m[l][r])

                X.append(temp)

                y.append(value)

            elif i<3 and j<6:

                for l in range(0,3):

                    for r in range(3,6):

                        if l!= i:

                            temp.append(m[l][r])

                        else:

                            if r!=j:

                                temp.append(m[l][r])

                X.append(temp)

                y.append(value)

            elif i<3 and j<9:

                for l in range(0,3):

                    for r in range(6,9):

                        if l!= i:

                            temp.append(m[l][r])

                        else:

                            if r!=j:

                                temp.append(m[l][r])

                X.append(temp)

                y.append(value)

                

            elif i<6 and j<3:

                for l in range(3,6):

                    for r in range(0,3):

                        if l!= i:

                            temp.append(m[l][r])

                        else:

                            if r!=j:

                                temp.append(m[l][r])

                X.append(temp)

                y.append(value)

            elif i<6 and j<6:

                for l in range(3,6):

                    for r in range(3,6):

                        if l!= i:

                            temp.append(m[l][r])

                        else:

                            if r!=j:

                                temp.append(m[l][r])

                X.append(temp)

                y.append(value)

            elif i<6 and j<9:

                for l in range(3,6):

                    for r in range(6,9):

                        if l!= i:

                            temp.append(m[l][r])

                        else:

                            if r!=j:

                                temp.append(m[l][r])

                X.append(temp)

                y.append(value)

                

            elif i<9 and j<3:

                for l in range(6,9):

                    for r in range(0,3):

                        if l!= i:

                            temp.append(m[l][r])

                        else:

                            if r!=j:

                                temp.append(m[l][r])

                X.append(temp)

                y.append(value)

            elif i<9 and j<6:

                for l in range(6,9):

                    for r in range(3,6):

                        if l!= i:

                            temp.append(m[l][r])

                        else:

                            if r!=j:

                                temp.append(m[l][r])

                X.append(temp)

                y.append(value)

            elif i<9 and j<9:

                for l in range(6,9):

                    for r in range(6,9):

                        if l!= i:

                            temp.append(m[l][r])

                        else:

                            if r!=j:

                                temp.append(m[l][r])

                X.append(temp)

                y.append(value)
X = np.asarray(X)

y = np.asarray(y)
#Shape of Input Array

X.shape
#Shape of output Array

y.shape
#Sample Input and Corresponding Output

print(X[0],y[0])
#Splitting data into training set and dev set

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)
#Training

from sklearn.ensemble import GradientBoostingClassifier as GBC

gbc = GBC(learning_rate = 0.1)

gbc.fit(X_train,y_train)
#Predictions

y_pred = gbc.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
#Confusion Matrix

print(cm)
#Visualisation of Confusion Matrix

#Code Source : https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix

import seaborn as sn

df_cm = pd.DataFrame(cm, index = [i for i in range(1,10)],

                  columns = [i for i in range(1,10)])

plt.figure(figsize = (20,10))

sn.set(font_scale=2)

sn.heatmap(df_cm, annot=True,annot_kws={"size": 15}, fmt = 'g')
#Accuracy

s = 0

for i in range(9):

    for j in range(9):

        if i==j:

            s += cm[i][j]
print('Accuracy : ',"{0:.2f}".format(s/X_test.shape[0]*100),'%')
#Predictor to function to preprocess data and return prediction

def pred(k,i,j):

            temp = []

            for l in range(i):

                temp.append(k[l][j])

            for l in range(i+1,9):

                temp.append(k[l][j])

            for l in range(j):

                temp.append(k[i][l])

            for l in range(j+1,9):

                temp.append(k[i][l])

            temp.append(i)

            temp.append(j)

            

            if i<3 and j<3:

                for l in range(0,3):

                    for r in range(0,3):

                        if l!= i:

                            temp.append(k[l][r])

                        else:

                            if r!=j:

                                temp.append(k[l][r])

               

            elif i<3 and j<6:

                for l in range(0,3):

                    for r in range(3,6):

                        if l!= i:

                            temp.append(k[l][r])

                        else:

                            if r!=j:

                                temp.append(k[l][r])

               

            elif i<3 and j<9:

                for l in range(0,3):

                    for r in range(6,9):

                        if l!= i:

                            temp.append(k[l][r])

                        else:

                            if r!=j:

                                temp.append(k[l][r])

                               

            elif i<6 and j<3:

                for l in range(3,6):

                    for r in range(0,3):

                        if l!= i:

                            temp.append(k[l][r])

                        else:

                            if r!=j:

                                temp.append(k[l][r])

                

            elif i<6 and j<6:

                for l in range(3,6):

                    for r in range(3,6):

                        if l!= i:

                            temp.append(k[l][r])

                        else:

                            if r!=j:

                                temp.append(k[l][r])

               

            elif i<6 and j<9:

                for l in range(3,6):

                    for r in range(6,9):

                        if l!= i:

                            temp.append(k[l][r])

                        else:

                            if r!=j:

                                temp.append(k[l][r])

                

                

            elif i<9 and j<3:

                for l in range(6,9):

                    for r in range(0,3):

                        if l!= i:

                            temp.append(k[l][r])

                        else:

                            if r!=j:

                                temp.append(k[l][r])

                

            elif i<9 and j<6:

                for l in range(6,9):

                    for r in range(3,6):

                        if l!= i:

                            temp.append(k[l][r])

                        else:

                            if r!=j:

                                temp.append(k[l][r])

                

            elif i<9 and j<9:

                for l in range(6,9):

                    for r in range(6,9):

                        if l!= i:

                            temp.append(k[l][r])

                        else:

                            if r!=j:

                                temp.append(k[l][r])

                

            temp = np.asarray(temp)

            temp = np.reshape(temp,(1,26))

            return gbc.predict(temp)
#Test Predictions for 4000 Puzzles(Ouput Not fitted Back to Input)

count = 0

acc = []

right_count = 0

blank = []

predicted = []

count_num = [0,0,0,0,0,0,0,0,0,0]

right = [0,0,0,0,0,0,0,0,0,0]

wrong = [0,0,0,0,0,0,0,0,0,0]

for p in range(1000,5000):

    count_arr = count

    right_arr = right_count

    y_pred = []

    k = quizzes[p]

    l = solutions[p]

    for i in range(9):

        tem = []

        for j in range(9):

            if k[i][j] == 0 and l[i][j] != 0:

                if l[i][j] == 1:

                    count_num[1]+=1

                elif l[i][j] == 2:

                    count_num[2]+=1

                elif l[i][j] == 3:

                    count_num[3]+=1

                elif l[i][j] == 4:

                    count_num[4]+=1

                elif l[i][j] == 5:

                    count_num[5]+=1

                elif l[i][j] == 6:

                    count_num[6]+=1

                elif l[i][j] == 7:

                    count_num[7]+=1

                elif l[i][j] == 8:

                    count_num[8]+=1

                elif l[i][j] == 9:

                    count_num[9]+=1

                count += 1

                prediction = pred(k,i,j)[0]

                if prediction == l[i][j]:

                    right_count += 1

                    if l[i][j] == 1:

                        right[1]+=1

                    elif l[i][j] == 2:

                        right[2]+=1

                    elif l[i][j] == 3:

                        right[3]+=1

                    elif l[i][j] == 4:

                        right[4]+=1

                    elif l[i][j] == 5:

                        right[5]+=1

                    elif l[i][j] == 6:

                        right[6]+=1

                    elif l[i][j] == 7:

                        right[7]+=1

                    elif l[i][j] == 8:

                        right[8]+=1

                    elif l[i][j] == 9:

                        right[9]+=1

                else:

                    if l[i][j] == 1:

                        wrong[1]+=1

                    elif l[i][j] == 2:

                        wrong[2]+=1

                    elif l[i][j] == 3:

                        wrong[3]+=1

                    elif l[i][j] == 4:

                        wrong[4]+=1

                    elif l[i][j] == 5:

                        wrong[5]+=1

                    elif l[i][j] == 6:

                        wrong[6]+=1

                    elif l[i][j] == 7:

                        wrong[7]+=1

                    elif l[i][j] == 8:

                        wrong[8]+=1

                    elif l[i][j] == 9:

                        wrong[9]+=1

                tem.append(prediction)

            else:

                tem.append(k[i][j])

        y_pred.append(tem)

    count_arr = count - count_arr

    right_arr = right_count - right_arr

    blank.append(count_arr)

    predicted.append(right_arr)

    acc.append((right_arr/count_arr)*100)

print(k)

print(l)

print(np.asarray(y_pred))

print('Predicted ',right_count,' out of ',count,' correctly.')

print('Accuracy : ','{0:.2f}'.format((right_count/count)*100),'%')

for i in range(1,10):

    print('Correct Predictions for ',i,' : ',right[i])

for i in range(1,10):

    print('Wrong Predictions for ',i,' : ',wrong[i])

for i in range(1,10):

    print('Accuracy for ',i,' : ','{0:.2f}'.format((right[i]/count_num[i])*100),'%')
#Plot of Accuracy for first 300 test arrays

plt.figure(figsize = (15,5))

sn.set_style('darkgrid')

plt.plot(acc[:300])

plt.title('Accuracy')
#Correct Predictions vs Number of blanks

plt.figure(figsize = (15,5))

plt.plot(blank[:300], color = 'red', label = 'No. of blanks')

plt.plot(predicted[:300], color = 'blue', label = 'Correct Predictions')

plt.legend()

plt.title('Accuracy')
#The Best and The Worst result

ma = 0

mi = 100

for i in range(len(acc)):

    if acc[i]>=ma:

        ma = acc[i]

        ma_i = 1000+i#Noting down index for future use

    if acc[i]<=mi:

        mi = acc[i]

        mi_i = 1000+i#Noting down index for future use

print('Highest accuracy attained is ',"{0:.2f}".format(ma),'%')

print('Lowest accuracy attained is ',"{0:.2f}".format(mi),'%')
print('Puzzle which gave highest accuracy(','{0:.2f}'.format(ma),'%)')

print(quizzes[ma_i])
#Preprocessing function

def loop(k,i,j):

            temp = []

            for l in range(i):

                temp.append(k[l][j])

            for l in range(i+1,9):

                temp.append(k[l][j])

            for l in range(j):

                temp.append(k[i][l])

            for l in range(j+1,9):

                temp.append(k[i][l])

            

            if i<3 and j<3:

                for l in range(0,3):

                    for r in range(0,3):

                        if l!= i:

                            temp.append(k[l][r])

                        else:

                            if r!=j:

                                temp.append(k[l][r])

            elif i<3 and j<6:

                for l in range(0,3):

                    for r in range(3,6):

                        if l!= i:

                            temp.append(k[l][r])

                        else:

                            if r!=j:

                                temp.append(k[l][r])

            

            elif i<3 and j<9:

                for l in range(0,3):

                    for r in range(6,9):

                        if l!= i:

                            temp.append(k[l][r])

                        else:

                            if r!=j:

                                temp.append(k[l][r])

                

                

            elif i<6 and j<3:

                for l in range(3,6):

                    for r in range(0,3):

                        if l!= i:

                            temp.append(k[l][r])

                        else:

                            if r!=j:

                                temp.append(k[l][r])

                

            elif i<6 and j<6:

                for l in range(3,6):

                    for r in range(3,6):

                        if l!= i:

                            temp.append(k[l][r])

                        else:

                            if r!=j:

                                temp.append(k[l][r])

                

            elif i<6 and j<9:

                for l in range(3,6):

                    for r in range(6,9):

                        if l!= i:

                            temp.append(k[l][r])

                        else:

                            if r!=j:

                                temp.append(k[l][r])

               

                

            elif i<9 and j<3:

                for l in range(6,9):

                    for r in range(0,3):

                        if l!= i:

                            temp.append(k[l][r])

                        else:

                            if r!=j:

                                temp.append(k[l][r])

                

            elif i<9 and j<6:

                for l in range(6,9):

                    for r in range(3,6):

                        if l!= i:

                            temp.append(k[l][r])

                        else:

                            if r!=j:

                                temp.append(k[l][r])

                

            elif i<9 and j<9:

                for l in range(6,9):

                    for r in range(6,9):

                        if l!= i:

                            temp.append(k[l][r])

                        else:

                            if r!=j:

                                temp.append(k[l][r])

                

            return temp
#Improving Accuracy by fitting output to input

for b in range(5):

    if b==0:

        new = np.copy(quizzes[ma_i])

    count = 0

    acc = []

    right_count = 0

    blank = []

    predicted = []

    count_num = [0,0,0,0,0,0,0,0,0,0]

    right = [0,0,0,0,0,0,0,0,0,0]

    wrong = [0,0,0,0,0,0,0,0,0,0]

    count_arr = count

    right_arr = right_count

    k = np.copy(new)

    l = solutions[ma_i]

    y_pred = np.copy(k)

    for i in range(9):

        tem = []

        for j in range(9):

            if k[i][j] == 0 or k[i][j] in loop(k,i,j):

                if l[i][j] == 1:

                    count_num[1]+=1

                elif l[i][j] == 2:

                    count_num[2]+=1

                elif l[i][j] == 3:

                    count_num[3]+=1

                elif l[i][j] == 4:

                    count_num[4]+=1

                elif l[i][j] == 5:

                    count_num[5]+=1

                elif l[i][j] == 6:

                    count_num[6]+=1

                elif l[i][j] == 7:

                    count_num[7]+=1

                elif l[i][j] == 8:

                    count_num[8]+=1

                elif l[i][j] == 9:

                    count_num[9]+=1

                

                count += 1

                prediction = pred(k,i,j)[0]

                loo = loop(k,i,j)

                #print(prediction,loo)

                if prediction not in loo:

                    tem.append(prediction)

                    k[i][j] = prediction

                    

                else:

                    tem.append(0)

                

                if prediction == l[i][j]:



                    right_count += 1

                    if l[i][j] == 1:

                        right[1]+=1

                    elif l[i][j] == 2:

                        right[2]+=1

                    elif l[i][j] == 3:

                        right[3]+=1

                    elif l[i][j] == 4:

                        right[4]+=1

                    elif l[i][j] == 5:

                        right[5]+=1

                    elif l[i][j] == 6:

                        right[6]+=1

                    elif l[i][j] == 7:

                        right[7]+=1

                    elif l[i][j] == 8:

                        right[8]+=1

                    elif l[i][j] == 9:

                        right[9]+=1

                else:

                    if l[i][j] == 1:

                        wrong[1]+=1

                    elif l[i][j] == 2:

                        wrong[2]+=1

                    elif l[i][j] == 3:

                        wrong[3]+=1

                    elif l[i][j] == 4:

                        wrong[4]+=1

                    elif l[i][j] == 5:

                        wrong[5]+=1

                    elif l[i][j] == 6:

                        wrong[6]+=1

                    elif l[i][j] == 7:

                        wrong[7]+=1

                    elif l[i][j] == 8:

                        wrong[8]+=1

                    elif l[i][j] == 9:

                        wrong[9]+=1

                    

            else:

                tem.append(k[i][j])

        y_pred[i] = np.copy(np.asarray(tem))

    new = np.copy(np.asarray(y_pred))
print('Predicted Array')

print(np.asarray(y_pred))

print('Correct Array')

print(l)
b = 0

for i in range(9):

    for j in range(9):

        if quizzes[ma_i][i][j] == 0:

            b +=1

print('No. of blanks in input : ',b)
c = 0

for i in range(9):

    for j in range(9):

        if y_pred[i][j] == l[i][j] and quizzes[ma_i][i][j] == 0:

            c += 1

print('Correct predictions = ', c)

print('Accuracy : ','{0:.2f}'.format((c/b)*100),'%')
print('Puzzle which gave lowest accuracy(','{0:.2f}'.format(mi),'%)')

print(quizzes[mi_i])
#Improving Accuracy by fitting output to input

for b in range(5):

    if b==0:

        new = np.copy(quizzes[mi_i])

    count = 0

    acc = []

    right_count = 0

    blank = []

    predicted = []

    count_num = [0,0,0,0,0,0,0,0,0,0]

    right = [0,0,0,0,0,0,0,0,0,0]

    wrong = [0,0,0,0,0,0,0,0,0,0]

    count_arr = count

    right_arr = right_count

    k = np.copy(new)

    l = solutions[mi_i]

    y_pred = np.copy(k)

    for i in range(9):

        tem = []

        for j in range(9):

            if k[i][j] == 0 or k[i][j] in loop(k,i,j):

                if l[i][j] == 1:

                    count_num[1]+=1

                elif l[i][j] == 2:

                    count_num[2]+=1

                elif l[i][j] == 3:

                    count_num[3]+=1

                elif l[i][j] == 4:

                    count_num[4]+=1

                elif l[i][j] == 5:

                    count_num[5]+=1

                elif l[i][j] == 6:

                    count_num[6]+=1

                elif l[i][j] == 7:

                    count_num[7]+=1

                elif l[i][j] == 8:

                    count_num[8]+=1

                elif l[i][j] == 9:

                    count_num[9]+=1

                count += 1

                prediction = pred(k,i,j)[0]

                loo = loop(k,i,j)

                #print(prediction,loo)

                if prediction not in loo:

                    tem.append(prediction)

                    k[i][j] = prediction

                else:

                    tem.append(0)

                if prediction == l[i][j]:



                    right_count += 1

                    if l[i][j] == 1:

                        right[1]+=1

                    elif l[i][j] == 2:

                        right[2]+=1

                    elif l[i][j] == 3:

                        right[3]+=1

                    elif l[i][j] == 4:

                        right[4]+=1

                    elif l[i][j] == 5:

                        right[5]+=1

                    elif l[i][j] == 6:

                        right[6]+=1

                    elif l[i][j] == 7:

                        right[7]+=1

                    elif l[i][j] == 8:

                        right[8]+=1

                    elif l[i][j] == 9:

                        right[9]+=1

                else:

                    if l[i][j] == 1:

                        wrong[1]+=1

                    elif l[i][j] == 2:

                        wrong[2]+=1

                    elif l[i][j] == 3:

                        wrong[3]+=1

                    elif l[i][j] == 4:

                        wrong[4]+=1

                    elif l[i][j] == 5:

                        wrong[5]+=1

                    elif l[i][j] == 6:

                        wrong[6]+=1

                    elif l[i][j] == 7:

                        wrong[7]+=1

                    elif l[i][j] == 8:

                        wrong[8]+=1

                    elif l[i][j] == 9:

                        wrong[9]+=1



            else:

                tem.append(k[i][j])

        y_pred[i] = np.copy(np.asarray(tem))

    new = np.copy(np.asarray(y_pred))
print('Predicted Array')

print(np.asarray(y_pred))

print('Correct Array')

print(l)
b = 0

for i in range(9):

    for j in range(9):

        if quizzes[mi_i][i][j] == 0:

            b +=1

print('No. of blanks in input : ',b)
c = 0

for i in range(9):

    for j in range(9):

        if y_pred[i][j] == l[i][j] and quizzes[mi_i][i][j] == 0:

            c += 1

print('Correct predictions = ', c)

print('Accuracy : ','{0:.2f}'.format((c/b)*100),'%')