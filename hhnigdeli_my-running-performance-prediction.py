import pandas as pd

import numpy as np

from datetime import datetime 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv('../input/run-20170920T174630.csv')

df.head(2)
df.info()
class clean(object):

    """ 

    Cleaning the running data

    """

    def __init__(self):

        self.df = pd.read_csv('../input/run-20170920T174630.csv')

        self.df = self.df.fillna(0)

        #print(self.df)

    def distance(self):

        cleanDistance=[]

        for i in range(0,len(df.distance)):

        #for i in range(0,2000):

            if self.df.distance[i] == 0:

                cleanDistance.append(self.df.distance[i+1])

            else:

                cleanDistance.append(self.df.distance[i])

        return(cleanDistance)

    def pace(self,t):

        z=[]

        cleanPace=[]

        for i in range(0,len(self.df.speed)):

        #for i in range(0,2000):

                a = 1 / (((self.df.speed[i])/1000)*60)

                z.append(a)

        for i in range(0,len(z)):

            if  z[i] > t :

                cleanPace.append(z[i-1])   

            else:

                cleanPace.append(z[i])

        return(cleanPace)

    def elevation(self):

        p=[]

        cleanElevation=[]

        for i in range(0,len(df.elevation)):

        #for i in range(0,2000):

            if self.df.elevation[i]  <= 0 and i>0:

                p.append(self.df.elevation[i-1])

            else:

                p.append(self.df.elevation[i])

        for i in range(0,len(p)):

            if p[i]  == 0:

                cleanElevation.append(p[i-1])

            else:

                cleanElevation.append(p[i])

        return(cleanElevation)

    def heartrate(self,x,y):

        """

        x = rework lower limit for HR data.

        y = new HR assingment after rework.

        """

        h=[]

        cleanHeartrate=[]

        for i in range(0,len(self.df.heartRate)):

        #for i in range(0,2000):

            if self.df.heartRate[i] < x:

                    h.append(y)

            else:

                    h.append(self.df.heartRate[i])

        for i in range(0,len(h)):

            if h[i] < x:

                    cleanHeartrate.append(y)

            else:

                    cleanHeartrate.append(h[i])

        return(cleanHeartrate)

    def climb(self):

        #print(str((len(df.elevation)/100)*0.5)+"Dk tahmini işlem süresi lütfen bekleyiniz....")

        climb=[]

        for i in range(0,len(df.elevation)):

        #for i in range(0,2000):

            #print(i)

            if i == 0:   

              climb.append(self.df.elevation[i+1]-self.df.elevation[i]) 

            else:

               climb.append(self.df.elevation[i]-self.df.elevation[i-1])    

        return(climb)

    def move(self):

        move=[]

        for i in range(0,len(df.distance)):

        #for i in range(0,2000):

            #print(i)

            if i == 0:

              move.append(self.df.distance[i+1]-self.df.distance[i])

            else:

              move.append(self.df.distance[i]-self.df.distance[i-1])   

        return(move)

    def clean(self,t=10,x=50,y=150):

        elev = self.elevation()

        pace = self.pace(t)

        heartrate = self.heartrate(x,y)

        dist = self.distance()

        clmb = self.climb()

        mv   = self.move()

        tm  = self.df.time

        cleanData         =    [elev,pace,dist,heartrate,clmb,mv,tm]

        cleanData         =    pd.DataFrame(np.transpose(cleanData))

        cleanData.columns =    ["elevation","pace","distance","heartrate","climb","move","time"]

        return (cleanData)

    def filtre(self,t=10,x=50,y=150,c=0.3,s=5.5):

        """

        (t=10,x=50,y=150,c=0.3,p=20,s=5.5)

        

        t = max pace

        

        x = rework lower limit for HR data.

        

        y = new HR assingment after rework.

        

        c =max and min climb limit.

        

        s = max move limit (mt/sn)

        

        """

        filtre1 = self.clean(t,x,y).pace != np.inf

        filtre2 = self.clean(t,x,y).climb <  c

        filtre3 = self.clean(t,x,y).climb >  -c

        filtre4 = self.clean(t,x,y).pace < t

        filtre5 = self.clean(t,x,y).move < s

        data = self.clean(t,x,y)

        return(data[filtre1&filtre2&filtre3&filtre4&filtre5])

    def graph(self,x=10,y=10):

        start_real = datetime.now()

        print("Process start:"+str(start_real))

        fig, axs = plt.subplots(2, 2,figsize=(x,y), sharex=True)

        fig.subplots_adjust(left=0.08, right=0.98, wspace=0.3)

        ax = axs[0, 0]

        ax.plot(self.filtre().elevation,color="green")

        ax.set_title('Elevation')

        ax = axs[0, 1]

        ax.plot(self.filtre().pace)

        ax.set_title('Pace')

        ax = axs[1, 0]

        ax.plot(self.filtre().heartrate,color="red" )

        ax.set_title('Heartrate')

        ax = axs[1, 1]

        ax.plot(self.filtre().distance)

        ax.set_title('Distance')

        stop_real = datetime.now()

        print("Process end :"+str(stop_real))

       
cl = clean()
cl.graph()
data = cl.filtre()
data.head(2)
data.describe()
from sklearn.cross_validation import train_test_split
X = data[["distance","heartrate","climb"]]
y = data[["pace"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=101 )

from sklearn.linear_model import LinearRegression
lm  = LinearRegression()
lm.fit(X_train,y_train)

print(lm.intercept_)
lm.coef_
X_train.columns
 

print(pd.DataFrame([{'distance':-4.64343296,"heartrate":-7.57638502,"climb":3.61995768,"intercept":7.31846705}]))

predictions = lm.predict(X_test)



predictions = pd.DataFrame(predictions)
predictions.coloumns= ["predictions"]
predictions = predictions.assign(y_test=y_test.values)
snsdata = predictions


snsdata.columns = ["predictions","y_test"]


sns.jointplot(x="y_test", y="predictions", data = snsdata,kind="kde", space=0, color="g")
sns.distplot(snsdata.y_test-snsdata.predictions,color="g")
np.mean(predictions)