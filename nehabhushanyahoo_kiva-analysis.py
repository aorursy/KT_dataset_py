import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
mpi_reg = pd.read_csv("../input/kiva_mpi_region_locations.csv")
loan = pd.read_csv("../input/kiva_loans.csv")
loanTheme_reg = pd.read_csv("../input/loan_themes_by_region.csv")
loanThem_id = pd.read_csv("../input/loan_theme_ids.csv")
type(loan)
#Rename partner_id in Loan_theme_id sheet
#loanThem_id.rename(columns={"Partner ID":"partner_id"},inplace= True)

#Rename partner_id in Loan_theme_reg sheet
loanTheme_reg.rename(columns={"Partner ID":"partner_id"},inplace=True)
loanTheme_reg.info()
#Merge Kiva_loan with LoanTheme_id sheet with unique parner id
new_loan = pd.merge(loan,loanThem_id,how="inner",on="id")

#merge new loan sheet to loan_themes_by_region.csv
new_loan2 = pd.merge(new_loan,loanTheme_reg,how='inner',on=["partner_id","Loan Theme ID","country","region"])

#Merge new_loan2, kiva_mpi_region_locations.csv 
new_loan_mpi = pd.merge(new_loan2,mpi_reg,how="inner",on=["lat","lon","ISO","country","region","LocationName"])


#Dropping columns that are repeated while merging with _y suffix
new_loan_mpi.drop(labels=["sector_y","Loan Theme Type_y","geo_y"],axis=1,inplace=True)
new_loan_mpi["Female"] = 0
new_loan_mpi["Male"] = 0
for i,val in enumerate(new_loan_mpi["borrower_genders"].values):
    b= val.split(", ")
    new_loan_mpi.loc[i,"Female"]= b.count('female')
    new_loan_mpi.loc[i,"Male"]= b.count('male')
    
#new_loan_mpi.info()
#determine column of type object and transform using one-hot encoder
#new_loan.head(4)
new_loan_mpi.shape
#new_loan_mpi.to_excel("MergedDoc.xlsx")
features = new_loan_mpi.iloc[:,2:43].values
features[1,:]

#Deleting repetative and dates column
features = np.delete(features,[0,3,5,7,9,10,11,14,15,17,18,21,23,24,27,28,30,31,35,38],axis= 1)

#removing texts and coordinates
features = np.delete(features,[13,15],axis=1)
np.size(features[0])
#Transform activity, sector, country_code, 
from sklearn.preprocessing import LabelEncoder
labelEn = LabelEncoder()

for i in range(0,np.size(features[0])-1):
    if features[:,i].dtype == 'object':
        features[:,i] = labelEn.fit_transform(features[:,i])
       
    
features[1,:]
from sklearn.cluster import KMeans

wcss =[]
x= 100
for i in range(1,x):
    kmeans = KMeans(n_clusters= i, init='k-means++', random_state=10)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)


plt.plot(range(1,x),wcss)
kmeans_model = KMeans(n_clusters= 8, init= "k-means++", random_state=10)
cluster = kmeans_model.fit_predict(features)
np.unique(cluster)
size_cluster = np.size(np.unique(cluster))
features[cluster==0]
for i in range(0,size_cluster-1):
    for j in range(0,size_cluster-1):
        s= 10
        plt.scatter(features[cluster == 0, i], features[cluster == 0, j], s = s, c = 'purple', label = 'Cluster 1')
        plt.scatter(features[cluster == 1, i], features[cluster == 1, j], s = s, c = 'yellow', label = 'Cluster 2')
        plt.scatter(features[cluster == 2, i], features[cluster == 2, j], s = s, c = 'green', label = 'Cluster 3')
        plt.scatter(features[cluster == 3, i], features[cluster == 3, j], s = s, c = 'cyan', label = 'Cluster 4')
        plt.scatter(features[cluster == 4, i], features[cluster == 4, j], s = s, c = 'magenta', label = 'Cluster 5')
        plt.scatter(features[cluster == 5, i], features[cluster == 5, j], s = s, c = 'gray', label = 'Cluster 6')
        plt.scatter(features[cluster == 6, i], features[cluster == 6, j], s = s, c = 'blue', label = 'Cluster 7')
        plt.scatter(features[cluster == 7, i], features[cluster == 7, j], s = s, c = 'red', label = 'Cluster 8')
        #plt.scatter(features[y_kmeans == 8,0], features[y_kmeans == 8, 1], s = 100, c = 'orange', label = 'Cluster 9')
        plt.scatter(kmeans_model.cluster_centers_[:, i], kmeans_model.cluster_centers_[:, j], s = 100, c = 'black', label = 'Centroids')
        text = 'Clusters of loans i:'+ str(i) + "and j:"+ str(j)
        plt.title(text)
        #plt.xlabel('Annual Income')
        #plt.ylabel('Spending Score (1-100)')
        plt.legend()
        plt.show()
sector = loan.groupby("sector")
sector_loanAmt = sector.funded_amount.sum()
sector_loanAmt = sector_loanAmt.sort_values(ascending=False)
sect