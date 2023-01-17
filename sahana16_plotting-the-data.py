from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

input_files =[]
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        input_files.append(os.path.join(dirname, filename))

import scipy.io
plots_name = {}
for index,inputs in enumerate(input_files):
    mat = scipy.io.loadmat(inputs)
    data = np.array(mat['image_cell'])
    plots_name [inputs] = np.mean(np.mean(mat['image_cell'],axis = 0),axis=0)
    if index%100 == 0:
        print("Awesome {} of the files have been processed".format(100*(index/len(input_files))))
    

    
plots_name
import pandas as pd
New_data = pd.DataFrame.from_dict(plots_name).T
New_data.values
values = np.array(New_data.values).reshape(-1,31)
### We cannot plot all the columns together, we will find 10 clusters in the data (random no )
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10)
predicted = kmeans.fit_predict(X=values)
New_data['Cluster'] = predicted

Groups = New_data.groupby(['Cluster'])
import matplotlib.pyplot as plt
x = [i for i in range(32)]
for cluster,dataframe in Groups:
    mean=dataframe.mean().values
    plt.plot(x,mean,label=str(cluster))
    
plt.legend()
plt.title("10 clusters of data")
plt.xlabel("Wavelength index")
plt.ylabel("Intensity")
plt.show()

