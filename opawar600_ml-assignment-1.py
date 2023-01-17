

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



        import math

from numpy import linalg as LA
df = pd.read_csv("/kaggle/input/normal/NormalSample.csv")

df.head(10)
#max value of x

df.describe()
from scipy.stats import iqr

inter_quartile_range = iqr(df.x)

N = df.x.count()

bin_width = 2*inter_quartile_range*(pow(N,-1/3)) #bin width = 2(IQR)N^(-1/3)

print(bin_width)
n = len(df.x)

min_x = df.x.min()

max_x = df.x.max()

print("Minimum = ",min_x,"\nMaximum = ",max_x)
#y = [0.4, 0.6, 0.7, 1.9, 2.4, 6.1, 6.2,7.3]

def get_midpoints(dfx,binw):

    x = dfx





    mini = math.floor(x.min())

    maxi = math.ceil(x.max())

    



    binw = binw     #Bin width



    midpoints = mini+binw/2



    no_of_bins = int((maxi-mini)/binw)

    print("Number of bins =",no_of_bins)

    start = mini+binw/2

    midps = [start]



    for i in range(0,no_of_bins-1):

        m = start+binw

        start = m

        midps.append(m)



    middf = pd.DataFrame(midps,columns = {"mi"}) #We get 100 midpoints and store it in a datframe

    return middf    #Midpoints
def density_estimate(midpoint):

    m1 = midpoint

    density = 0

    for i in df.x:

        u = (i-m1)/binw

        if (u>-0.5 and u<=0.5):

            density+=1

    return density/(n*binw)
def answer(df,binwidth):

    x = df.x

    binw = binwidth

    middf = get_midpoints(df.x,binw)

    lis = []

    for m in middf.mi:

        lis.append(density_estimate(m))



    middf["p(mi)"] = np.array(lis)

    #plt.step(middf.mi,middf["p(mi)"])

    return middf
binw = 2

ans_df2 = answer(df,binw)



plt.hist(x=df.x,range=(26,36), bins = 5,color = "dimgray") # h = 2

plt.title("Bin Width = 2")

plt.grid(axis='y', alpha=0.2)

plt.savefig("h2.png")

ans_df2
binw = 1

ans_df1 = answer(df,binw)

print(ans_df1)

plt.hist(x=df.x,range=(26,36), bins = 10,color = "dimgray") # h = 1

plt.title("Bin Width = 1")

plt.grid(axis='y', alpha=0.2)

plt.savefig("h1.png")

ans_df1
binw = 0.5

ans_df05 = answer (df,binw)

print("Co-ordinates of density \n",ans_df05)

plt.hist(x=df.x,range=(26,36), bins = 20,color = 'dimgray') # h = 0.5

plt.title("Bin Width = 0.5")

plt.grid(axis='y', alpha=0.2)

plt.savefig("h05.png")

ans_df05
binw = 0.1

ans_df01 = answer (df,binw)

ans_df01

plt.hist(x=df.x,range=(26,36), bins = 100,color = "dimgrey") # h = 2

plt.title("Bin Width = 0.1")

plt.grid(axis='y', alpha=0.2)

plt.savefig("h01.png")

ans_df01
import seaborn as sns

plt.boxplot(df.x,vert = False)

plt.title("Boxplot of X")

plt.savefig("boxplot_x.png")
from scipy.stats import iqr

inter_quartile_range = iqr(df.x)

q1 = np.percentile(df.x,25)

q3 = np.percentile(df.x,75)

l_whisker = q1 - 1.5*inter_quartile_range

u_whisker = q3 + 1.5*inter_quartile_range

print ("Lower Whisker = ",l_whisker,"\nUpper Whisker = ",u_whisker)
zero = []

ones = []

for i in range(0,df.x.count()):

    if df.group[i] == 0:

        zero.append(df.x[i])

    else:

        ones.append(df.x[i])

zeros = np.array(zero)

ones = np.array(ones)
from scipy.stats import iqr

inter_quartile_range = iqr(zeros)

q1 = np.percentile(zeros,25)

q3 = np.percentile(zeros,75)

l_whisker = q1 - 1.5*inter_quartile_range

u_whisker = q3 + 1.5*inter_quartile_range

print ("Lower Whisker = ",l_whisker,"\nUpper Whisker = ",u_whisker)
from scipy.stats import iqr

inter_quartile_range = iqr(ones)

q1 = np.percentile(ones,25)

q3 = np.percentile(ones,75)

l_whisker = q1 - 1.5*inter_quartile_range

u_whisker = q3 + 1.5*inter_quartile_range

print ("Lower Whisker = ",l_whisker,"\nUpper Whisker = ",u_whisker)
plt.boxplot([ones,zero,df.x],widths = 0.4)

plt.xticks(np.arange(1,4),("Ones","Zeros","X"))

plt.title("BoxPlot For Each Category")

plt.savefig("CategoryBoxplot.png")
df0 = df[df['group']==0]

df0.describe()
df1 = df[df['group']==1]

df1.describe()
fraud_df = pd.read_csv("/kaggle/input/fraudknn/Fraud.csv")
fraud_df.head()
num_no_frauds = fraud_df.FRAUD.value_counts()[0]

num_of_frauds = fraud_df.FRAUD.value_counts()[1]

count = fraud_df.FRAUD.count()
fraud_percent = num_of_frauds/count*100

print(round(fraud_percent,4))
fraud_df.columns
sns.boxplot(data = fraud_df , x = 'TOTAL_SPEND' , y = 'FRAUD',orient = 'h')

plt.title("Total Spent")

plt.savefig("TotalSpent.png")
sns.boxplot(data = fraud_df , x = 'NUM_MEMBERS' , y = 'FRAUD',orient = 'h')

plt.title("Number of members covered")

plt.savefig("Num_members.png")
sns.boxplot(data = fraud_df , x = 'MEMBER_DURATION' , y = 'FRAUD', orient = 'h')

plt.title("Member Duration")

plt.savefig("Mem_duration.png")
sns.boxplot(data = fraud_df , x = 'OPTOM_PRESC' , y = 'FRAUD' , orient = 'h')

plt.title("Number of optical examinations")

plt.savefig("Optom_Presc.png")
sns.boxplot(data = fraud_df , x = 'DOCTOR_VISITS' , y = 'FRAUD' ,orient = 'h')

plt.title("Number of visits to a doctor  ")

plt.savefig("Doctor_Visit.png")
sns.boxplot(data = fraud_df , x = 'NUM_CLAIMS' , y = 'FRAUD',orient = 'h')

plt.title("Number of claims made recently ")

plt.savefig("Num_claims.png")
fraud_df.head()

fraud_df = pd.read_csv("/kaggle/input/fraudknn/Fraud.csv",usecols=["TOTAL_SPEND","DOCTOR_VISITS","NUM_CLAIMS","MEMBER_DURATION","OPTOM_PRESC","NUM_MEMBERS"])
x = np.matrix(fraud_df)
xtx = x.transpose() * x

print("t(x) * x = \n", xtx)



# Eigenvalue decomposition

evals, evecs = LA.eigh(xtx)

print("Eigenvalues of x = \n", evals)

print("Eigenvectors of x = \n",evecs)



# Here is the transformation matrix

transf = evecs * LA.inv(np.sqrt(np.diagflat(evals)));

print("Transformation Matrix = \n", transf)



# Here is the transformed X

transf_x = x * transf;

print("The Transformed x = \n", transf_x)

# Check columns of transformed X

xtx = transf_x.transpose() * transf_x;

print("Expect an Identity Matrix = \n", xtx)



# Orthonormalize using the orth function 

import scipy

from scipy import linalg as LA2



orthx = LA2.orth(x)

print("The orthonormalize x = \n", orthx)



# Check columns of the ORTH function

check = orthx.transpose().dot(orthx)

print("Also Expect an Identity Matrix = \n", check)
from sklearn.neighbors import KNeighborsClassifier
target = pd.read_csv("/kaggle/input/fraudknn/Fraud.csv",usecols=["FRAUD"])
from sklearn import metrics

knn = KNeighborsClassifier(n_neighbors=5, metric = "euclidean")

knn.fit(transf_x,target)

predictions = knn.predict(transf_x)

print(metrics.accuracy_score(target,predictions))
test = [[7500,15,3,127,2,2]] * transf;

test
neighs = knn.kneighbors(test,return_distance=False)

neighs
fraudulent_df = pd.read_csv("/kaggle/input/fraudknn/Fraud.csv")

fraudulent_df.iloc[neighs[0][0:]]