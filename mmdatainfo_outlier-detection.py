import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# This magic will ensure that the plots are in Jupyter in high quality
# see https://stackoverflow.com/questions/25412513/inline-images-have-low-quality
%config InlineBackend.figure_format = "svg"
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
def boxplotbased(vecin,whis=1.5):
    """
    Same as using PyPlot: 
    out = plt.boxplot(vecin,whis=whis); 
    out = out[0].get_data()[1];
    plt.close()
    """
    Q = np.quantile(vecin,[0.25,0.75]);
    Qmax = Q[1] + whis*(Q[1]-Q[0]);
    Qmin = Q[0] - whis*(Q[1]-Q[0]);
    out = np.ones(vecin.shape[0]);
    out[vecin<Qmin] = -1;
    out[vecin>Qmax] = -1;
    return out
model_name = ["Isolation Forest","Mimimum Covariance", "Local Outlier Factor", "Boxplot/stat Based"]
model_def = [IsolationForest(contamination=0.03,behaviour ="new"),
             EllipticEnvelope(contamination=0.03), 
             LocalOutlierFactor(contamination="auto"),
             boxplotbased]
def genpolydata(X):
    return (10.0+3.0*X[:,0]-2.0*X[:,1]+0.1*(X[:,0]**2)+0.4*(X[:,1]**2)).reshape(X.shape[0],1);
# regular grid converted to features matrix
x0, y0 = np.meshgrid(np.linspace(0.0,200.0,num=30), np.linspace(-100,100,num=20))

# Convert to feature matrix
X = np.concatenate((np.asanyarray(x0).reshape(x0.size,1),
                    np.asanyarray(y0).reshape(y0.size,1)),
                   axis=1);
X = np.concatenate((X,genpolydata(X)),
                   axis=1);
np.random.seed(0)
X[:,-1] += np.random.randn(X.shape[0])*100; # noise
for i in range(0,X.shape[0]):
    if i < 5:
        X[i,-1] = 10000; # outlier
def plotdata():
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x0,y0,genpolydata(X).reshape(x0.shape));
    ax.scatter(X[:,0],X[:,1],X[:,2],c="g",marker=".")
    return ax;
ax = plotdata();
for (n,m,c,s) in zip(model_name,model_def,["ro","kv","mx","bo"],[50,30,20,10]):
    if n != "Boxplot/stat Based":
        xi = m.fit_predict(X[:,2].reshape(X.shape[0],1));
#         xi = m.fit_predict(X) # 
    else:
        xi = m(X[:,2]);
    ax.scatter(X[xi==-1,0],X[xi==-1,1],X[xi==-1,2],
               c=c[0],marker=c[1],s=s);
    print("{}: \tnumber of outliers identified = {}".format(n,xi[xi==-1].size))
# Get -1 | 1 vector ID (standard sklearn format)
out_id = boxplotbased(X[:,2]);
Xrem = X[out_id==1,:]
from sklearn.preprocessing import MinMaxScaler
lof = LocalOutlierFactor(contamination="auto").fit(X);
tag = lof.negative_outlier_factor_ # sklearn return inverted LOF
tag = MinMaxScaler().fit_transform(tag.reshape(X.shape[0],1));
plt.figure(figsize=(8,3));
plt.plot(tag,"k.")
plt.title("LOF-based outlier tag (close to outlier=zero)");
# Just show example for SVM classification
from sklearn.svm import SVC
# Try to classify all 5 outliers
y = out_id
# Add Tag to feature matrix. In this case, same result would be obtained even without the Tag
Xc = np.concatenate((X,tag.reshape(X.shape[0],1)),axis=1);
svc = SVC(gamma="scale").fit(Xc,y)
# Count number of classified outliers
(svc.predict(Xc)[svc.predict(Xc)==-1]).size
