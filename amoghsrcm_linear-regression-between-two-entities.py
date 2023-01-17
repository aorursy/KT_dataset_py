
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

def Ack(): # Acknowledgement about the code
    print()
    print("\n Data visualization with linear regression between two entities")
    print(" Data source : https://www.kaggle.com/kiva/data-science-for-good-kiva-crowdfunding/data")
    print()

def inpo(ofil,e1,e2): #deriving 2 Entities
    dataset = ofil
    E1=dataset[e1]
    E2=dataset[e2]
    return E1, E2

def convo(lwn,trm): #converting Entities into np arrays
    import numpy as np # linear algebra
    np.set_printoptions(threshold=np.nan)
    x = np.array(lwn).reshape(-1, 1)
    y = np.array(trm)
    return x,y

def xtyt(xx,yy): #spliting xtrain xtest ytrain ytest
    from sklearn.cross_validation import train_test_split 
    xtrain, xtest, ytrain, ytest = train_test_split(xx,yy,test_size=1/3, random_state=0)
    return xtrain, xtest, ytrain, ytest

def regr(xxtr,yytr,xxte): #linear regression
    from sklearn.linear_model import LinearRegression 
    regressor = LinearRegression()
    regressor.fit(xxtr, yytr)
    pred = regressor.predict(xxte)
    return regressor

def ttRv(x1tr,y1tr,x1te,xlbl,ylbl): #visualization of training dataset
    import matplotlib.pyplot as plt 
    regs=regr(x1tr,y1tr,x1te)
    plt.scatter(x1tr, y1tr, color= 'red')
    plt.plot(x1tr, regs.predict(x1tr), color = 'blue')
    plt.title ("Visuals for Training Dataset")
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.show()

def tRv(x2tr,y2tr,x2te,y2te,xlbl,ylbl): #visualization of test dataset
    import matplotlib.pyplot as plt 
    regs=regr(x2tr,y2tr,x2te)
    plt.scatter(x2te, y2te, color= 'red')
    plt.plot(x2tr, regs.predict(x2tr), color = 'blue')
    plt.title("Visuals for Test DataSet")
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.show()

def Opn(file):  # Opens csv file in read mode
    import pandas as pnd # data processing, CSV file I/O (e.g. pd.read_csv)
    return pnd.read_csv(file)

def menu(): #main module to give specifications
    
    import time
    print()
    print(" Execution started at :-\n",time.asctime(time.localtime(time.time())))
    a=time.asctime(time.localtime(time.time()))
    aa=(int(a[14])*10+int(a[15]))*60+int(a[17])*10+int(a[18])
    
    Ack()
    ofile = Opn('../input/kiva_loans.csv')
    lo,tr = inpo(ofile,'loan_amount','term_in_months')
    x1,y1 = convo(lo,tr)
    xtr,xte,ytr,yte = xtyt(x1,y1)
    ttRv(xtr,ytr,xte,"loan","term")
    tRv(xtr,ytr,xte,yte,"loan","term")

    print()
    print(" Execution completed at :-\n",time.asctime(time.localtime(time.time())))
    b=time.asctime(time.localtime(time.time()))
    bb=(int(b[14])*10+int(b[15]))*60+int(b[17])*10+int(b[18])

    print()
    print(" Total time taken for execution in seconds \t:\t ",bb-aa)
    
menu()
