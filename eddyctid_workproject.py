# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import glob

import pandas as pd

import numpy as np



def rpm2cmps(x):

    return ((x*0.0326*0.0417));







def createSheet(kmph,timeStep):

    mps = kmph/3.6;

    rpmSpeed = (mps/0.0326)/0.0417;

    startDistance = 255;

    distanceDecrement = mps*100/100;

    #print(mps);

    #print(rpmSpeed);

    #print(distanceDecrement);

    

    #timeStep Calculation

    t = np.arange(0,timeStep,timeStep/1000);

    #print(t);

    #print(type(t));

    #np.savetxt('newData1.csv',t,delimiter=",");

    

    #rpmInputs

    counter = 0;

    distanceList = [];

    rpmList = [];

    

    while counter < 9:

        for i in range(0,10,1):

            startDistance = startDistance - rpm2cmps(i);

            #print(startDistance);

            distanceList.append(startDistance);

            rpmList.append(i);

            counter = counter+1;

            #print(distanceList);

        

    while startDistance > 0:

        startDistance = startDistance - distanceDecrement;

        distanceList.append(startDistance);

        rpmList.append(rpmSpeed);

    

    

    dict = {'t': t,'Distance':distanceList,'RPM':rpmList};

    df = pd.DataFrame.from_dict(dict,orient='index');

    finalResult = df.transpose();

    

    

    filename = "Input.csv";

    

    fileExists = glob.glob(filename)

    if not fileExists:

        finalResult.to_csv("../Input.csv");

        print("Done");

    else:

        print ("File already exists");

    

    return;

    

    
createSheet(10,4)