import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
import matplotlib.pyplot as plt

breast_cc = datasets.load_breast_cancer()
breast_cc.keys()
def outlier_wrt_IQR(x):
    """
    This function is used in case the features is univariate!!
    input_args: 
        x (list of the values)
    Returns: The outlier of x 
    """
    ## initialize
    outliers = []
    ## basic statistics
    df = pd.DataFrame({'x': x})
    mi = df.min().values
    ma = df.max().values
    Q3 = df.describe().loc['75%'].values
    Q1 = df.describe().loc['25%'].values
    IQR = Q3- Q1
    
    ## The outlier is not in the interval [max{min(x), Q1 - 1.5*IQR}, min{max(x), Q3 + 1.5*IQR}]
    lower = max(mi, Q1 - 1.5*IQR)
    upper = min(ma, Q3 + 1.5*IQR)
    for t in x:
        if (lower > t) | (t > upper):
            outliers.append(t)
    outliers.sort()
    return outliers
X = breast_cc['data']
X9 = X[:, 8]
df = pd.DataFrame({'x': X9})
sns.boxplot(X9)
df.describe().T
print(outlier_wrt_IQR(X9))