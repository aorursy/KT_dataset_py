import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import math
import re
from matplotlib.pyplot import figure
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
data = pd.read_csv('../input/multipleChoiceResponses.csv', low_memory=False)
gender = data['Q1']

gCopy = gender.copy()

female_count = gCopy.str.count('Female').sum()
male_count = gCopy.str.count('Male').sum()
other_count = len(gCopy)-(female_count+male_count)

barcount = [  other_count/len(gCopy),female_count/len(gCopy),male_count/len(gCopy)]
barinfo = [ 'Other','Female','Male']
y_pos = np.arange(len(barinfo))
total = female_count+ male_count + other_count
# Create horizontal bars
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.barh(y_pos, barcount, color=[(0.6, 0.3, 0.65, 0.8),(1, 0.2, 0.7, 1),(0.2, 0.4, 0.6, 0.6) ],
         edgecolor=['purple','purple','blue'])
plt.title('Percentage of genders doing this survey')
plt.yticks(y_pos, barinfo)
ax.set_facecolor('lavender')
plt.grid( linestyle='-', linewidth=0.3, color='white')
plt.show()
print('there is a total of', female_count, 'females, around', 100*female_count/total, '%')
print('there is a total of', male_count, 'males, around', 100*male_count/total, '%')
print('there is a total of', other_count, 'who did not choose or prefer not to choose gender, around', 100*other_count/total,'%')
def organizeTitle (fi,mi,oi,title):
    titles = []
    titles.append(title[1])
    for i in range(1,len(title)):
        c = 0
        for j in range(0,len(titles)):
            if(titles[j]== title[i]):
                c =1
        if(c==0):
            if(title[i]!='0'):
                titles.append(title[i])


    titles = np.asarray(titles)
    resultado = []
    resultado2 = []
    s = title[fi]
    m = np.where(title[fi]=='Data Scientist')

    for i in range(0,len(titles)):
        f = len(title[np.where(title[fi]==titles[i])])
        m = len(title[np.where(title[mi]== titles[i])])
        o = len(title[np.where(title[oi]== titles[i])])
        total = f+m+o
        if(titles[i]!='0'):
            c1 = [titles[i],100*f/total,100* m/total, 100*o/total]
            c2 = [titles[i],100*f/(len(title[fi])),100* m/(len(title[mi])),100* o/len(title[oi])]
            resultado.append(c1)
            resultado2.append(c2)
        
    resultado = np.asarray(resultado)
    resultado2 = np.asarray(resultado2)
    return resultado, resultado2, titles

def plot1(r1,r2,t,mess):
    # plot
    barWidth = 0.85
    names = t
    orangeBars = list(r1[:,1])

    blueBars = list(r1[:,2])
    greenBars = list(r1[:,3])
    r = list(np.linspace(0,len(greenBars),len(greenBars)))

    for i in range(0,len(orangeBars)):
        orangeBars[i] = float(orangeBars[i])
        blueBars[i] = float(blueBars[i])
        greenBars[i] = float(greenBars[i])

    # Create green Bars
    plt.bar(r, greenBars, color='#b5ffb9', edgecolor='white', width=barWidth)
    # Create orange Bars
    plt.bar(r, orangeBars, bottom=greenBars, color='#f9bc86', edgecolor='white', width=barWidth)
    # Create blue Bars
    plt.bar(r, blueBars, bottom=[i+j for i,j in zip(greenBars, orangeBars)], color='#a3acff', edgecolor='white', width=barWidth)

    # Custom x axis
    plt.rcParams["figure.figsize"] = (15,4)
    plt.xticks(r, names)
    plt.xlabel("titles")
    plt.xticks(rotation=90)
    plt.title(mess)
    # Show graphic
    plt.show()
def plot2(r1,r2,t,mess):
    barwidth = 0.3
    names = t
    orangeBars = list(r2[:,1])
    blueBars = list(r2[:,2])
    greenBars = list(r2[:,3])
    r = list(np.linspace(0,len(greenBars),len(greenBars)))
    rr2 = list(np.linspace(r[0]+barwidth,r[len(r)-1]+barwidth,len(greenBars)))
    rr3 = list(np.linspace(rr2[0]+barwidth,rr2[len(rr2)-1]+barwidth,len(greenBars)))

    for i in range(0,len(orangeBars)):
        orangeBars[i] = float(orangeBars[i])
        blueBars[i] = float(blueBars[i])
        greenBars[i] = float(greenBars[i])
    
    plt.rcParams["figure.figsize"] = (15,5)
    plt.bar(r, orangeBars, width = barwidth, color = '#f9bc86', edgecolor = 'white', label='female')

    plt.bar(rr2, blueBars, width = barwidth, color = '#a3acff', edgecolor = 'white',  label='male')

    plt.bar(rr3, greenBars, width = barwidth, color = '#b5ffb9', edgecolor = 'white',  label='other')

    plt.ylabel('porcentage')
    plt.legend()
    plt.xticks(rr2,names)
    plt.xticks(rotation=90)
    plt.title(mess)
    # Show graphic
    plt.show()

title = data['Q2']
title.fillna(0, inplace=True)
title = np.asarray(title)
gCopy = np.asarray(gCopy)
mess = 'Age'

female_i = np.where(gCopy == 'Female')
male_i = np.where(gCopy == 'Male')
other_i = np.where(gCopy[np.where(gCopy!='Male')]!= 'Female')

r1, r2, t = organizeTitle(female_i,male_i,other_i,title)

plot2(r1,r2,t,mess)


title = data['Q6']
title.fillna(0, inplace=True)
title = np.asarray(title)
gCopy = np.asarray(gCopy)

mess = "Occupation"
r1, r2, t = organizeTitle (female_i,male_i,other_i,title)

plot1(r1,r2,t,mess)
plot2(r1,r2,t,mess)

title = data['Q4']
title.fillna(0, inplace=True)
title = np.asarray(title)
gCopy = np.asarray(gCopy)
mess = 'Studies'
r1, r2, t = organizeTitle (female_i,male_i,other_i,title)

plot1(r1,r2,t,mess)
plot2(r1,r2,t,mess)
title = data['Q9']
title.fillna(0, inplace=True)
title = np.asarray(title)
gCopy = np.asarray(gCopy)
mess = 'Annual salary'
r1, r2, t = organizeTitle(female_i,male_i,other_i,title)
plot1(r1,r2,t,mess)
plot2(r1,r2,t,mess)
def organizar(x, y):
    for i in range(len(x)):
        swap = i + np.argmin(x[i:])
        (x[i], x[swap]) = (x[swap], x[i])
        (y[i], y[swap]) = (y[swap], y[i])
    return x, y

def plotx1x2(f,m,o,x1,x2,mess):
    
    fx1 = x1[f]
    mx1 = x1[m]
    ox1 = x1[o]
    
    fx2 = x2[f]
    mx2 = x2[m]
    ox2 = x2[o]
    
    fx11 = []
    fx22 = []
    mx11 = []
    mx22 = []
    ox11 = []
    ox22 = []
    
    for i in range(0, len(fx1)):
        if(fx1[i]!=0):
            if(fx2[i]!=0):
                if(fx2[i].find("do") == -1):
                    if(fx1[i].find("+")!=-1):
                        ff2 = float(re.split('[- +]',fx2[i])[0].replace(',',''))
                        if(ff2<500000):
                            fx11.append( float(fx1[i].split(' ')[0]))
                            fx22.append(ff2*10000+10000)
                    else:
                        ff2 = float(re.split('[- +]',fx2[i])[0].replace(',',''))
                        if(ff2<500000):
                            fx11.append( float(fx1[i].split('-')[0]))
                            fx22.append(ff2*10000+10000)
        else:
            if(fx2[i]!=0):
                if(fx2[i].find("do") == -1):
                    ff2 = float(re.split('[- +]',fx2[i])[0].replace(',',''))
                    if(ff2<500000):
                        fx11.append( float(fx1[i]))
                        fx22.append(ff2*10000+10000)
                    
    for i in range(0, len(mx1)):
        if(mx1[i]!=0):
            if(mx2[i]!=0):
                if(mx2[i].find("do") == -1):
                    if(mx1[i].find("+")!=-1):

                        ff2 = float(re.split('[- +]',mx2[i])[0].replace(',',''))
                        if(ff2<500000):
                            mx11.append( float(mx1[i].split(' ')[0]))
                            mx22.append(ff2*10000+10000)
                    else:
                        
                        ff2 = float(re.split('[- +]',mx2[i])[0].replace(',',''))
                        if(ff2<500000):
                            mx11.append( float(mx1[i].split('-')[0]))
                            mx22.append(ff2*10000+10000)
        else:
            if(mx2[i]!=0):
                if(mx2[i].find("do") == -1):
                    
                    ff2 = float(re.split('[- +]',mx2[i])[0].replace(',',''))
                    if(ff2<500000):
                        mx11.append( float(mx1[i]))
                        mx22.append(ff2*10000+10000)
                    
    for i in range(0, len(ox1)):
        if(ox1[i]!=0):
            if(ox2[i]!=0):
                if(ox1[i].find("How") == -1 and ox2[i].find('I')==-1):
                    if(ox1[i].find("+")!=-1):
                        ox11.append( float(ox1[i].split(' ')[0]))
                        ox22.append(float(re.split('[- +]',ox2[i])[0].replace(',',''))*10000+10000)
                    else:
                        ox11.append( float(ox1[i].split('-')[0].replace(',','')))
                        ox22.append(float(re.split('[- +]',ox2[i])[0].replace(',',''))*10000+10000)
        else:
            if(ox2[i]!=0):
                if(ox2[i].find("I") == -1):
                    ox11.append( float(ox1[i]))
                    ox22.append(float(re.split('[- +]',ox2[i])[0].replace(',',''))*10000+10000)
    
    l =[]
    l.append(fx11[0])
    for i in range(0,len(fx11)):
        c = 0
        for j in range(0,len(l)):
            if(l[j]== fx11[i]):
                c =1
        if(c==0):
            if(title[i]!='0'):
                l.append(fx11[i])   
    fl =[]
    ml =[]
    ol =[]
    fx11 = np.asarray(fx11)
    fx22 = np.asarray(fx22)
    mx11 = np.asarray(mx11)
    mx22 = np.asarray(mx22)
    ox11 = np.asarray(ox11)
    ox22 = np.asarray(ox22)
    
    for i in range(0, len(l)):
        
        sfl = np.where(fx11==l[i])
        fl.append(np.mean(fx22[sfl]))
        ml.append(np.mean(mx22[np.where(mx11==l[i])]))
        ol.append(np.mean(ox22[np.where(ox11==l[i])]))
        
    plt.style.use('seaborn-darkgrid')
    plt.rcParams["figure.figsize"] = (15,10)
    plt.subplot(2,2,1)
    plt.scatter(fx11,fx22)
    x,y = organizar(l,fl)
    plt.plot(x,y,'red')
    plt.subplot(2,2,2)
    plt.scatter(mx11,mx22)
    x,y = organizar(l,ml)
    plt.plot(x,y,'red')
    plt.subplot(2,2,3)
    plt.scatter(ox11,ox22)
    x,y = organizar(l,ol)
    plt.plot(x,y,'red')
    xx = np.concatenate((fx11,mx11), axis=0)
    xx = np.concatenate((xx, ox11), axis=0)
    yy = np.concatenate((fx22,mx22), axis=0)
    yy = np.concatenate((yy, ox22), axis=0)
    
    file1 = np.linspace(0,0,len(fx11))
    file2 = np.linspace(1,1, len(mx11))
    file3 = np.linspace(2,2, len(ox11))
    file = np.concatenate((file1,file2), axis=0)
    file = np.concatenate((file, file3), axis=0)
    return xx, yy, file

salary = data['Q9']
salary.fillna(0, inplace=True)
salary = np.asarray(salary)

yearsexp = data['Q8']
yearsexp.fillna(0, inplace=True)
yearsexp = np.asarray(yearsexp)


xx, yy, dat2 = plotx1x2(female_i,male_i,other_i,yearsexp,salary,'title')
dat = []
for i in range(0,len(xx)):
    dat.append([xx[i], yy[i]])
    
dat = np.asarray(dat)
kf = KFold(n_splits=4)

cc =0
for train_index, test_index in kf.split(dat):
    X_train, X_test = dat[train_index], dat[test_index]
    y_train, y_test = dat2[train_index], dat2[test_index]
    clf = LogisticRegression(random_state=0, solver= 'lbfgs', multi_class = 'multinomial').fit(X_train, y_train)
    predictors = clf.predict_proba(X_test)
    if(cc==0):
        plt.figure()
        plt.scatter(X_test,predictors)
        plt.show()
        cc =1

