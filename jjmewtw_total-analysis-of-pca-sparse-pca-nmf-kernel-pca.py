import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from mlxtend.preprocessing import minmax_scaling

import seaborn as sns

from sklearn.decomposition import PCA,SparsePCA,KernelPCA,NMF

from sklearn.datasets import make_circles
summer_products_path = "../input/summer-products-and-sales-in-ecommerce-wish/summer-products-with-rating-and-performance_2020-08.csv"

unique_categories_path = "../input/summer-products-and-sales-in-ecommerce-wish/unique-categories.csv"

unique_categories_sort_path = "../input/summer-products-and-sales-in-ecommerce-wish/unique-categories.sorted-by-count.csv"



summer_products = pd.read_csv(summer_products_path)

unique_categories = pd.read_csv(unique_categories_path)

unique_categories_sort = pd.read_csv(unique_categories_sort_path)



df = summer_products



C = (df.dtypes == 'object')

CategoricalVariables = list(C[C].index)

Integer = (df.dtypes == 'int64') 

Float   = (df.dtypes == 'float64') 

NumericVariables = list(Integer[Integer].index) + list(Float[Float].index)



df[NumericVariables]=df[NumericVariables].fillna(0)

df=df.drop('has_urgency_banner', axis=1) # 70 % NA's



df[CategoricalVariables]=df[CategoricalVariables].fillna('Unknown')

df=df.drop('urgency_text', axis=1) # 70 % NA's

df=df.drop('merchant_profile_picture', axis=1) # 86 % NA's



C = (df.dtypes == 'object')

CategoricalVariables = list(C[C].index)

Integer = (df.dtypes == 'int64') 

Float   = (df.dtypes == 'float64') 

NumericVariables = list(Integer[Integer].index) + list(Float[Float].index)



Size_map  = {'NaN':1, 'XXXS':2,'Size-XXXS':2,'SIZE XXXS':2,'XXS':3,'Size-XXS':3,'SIZE XXS':3,

            'XS':4,'Size-XS':4,'SIZE XS':4,'s':5,'S':5,'Size-S':5,'SIZE S':5,

            'M':6,'Size-M':6,'SIZE M':6,'32/L':7,'L.':7,'L':7,'SizeL':7,'SIZE L':7,

            'XL':8,'Size-XL':8,'SIZE XL':8,'XXL':9,'SizeXXL':9,'SIZE XXL':9,'2XL':9,

            'XXXL':10,'Size-XXXL':10,'SIZE XXXL':10,'3XL':10,'4XL':10,'5XL':10}



df['product_variation_size_id'] = df['product_variation_size_id'].map(Size_map)

df['product_variation_size_id']=df['product_variation_size_id'].fillna(1)

OrdinalVariables = ['product_variation_size_id']



Color_map  = {'NaN':'Unknown','Black':'black','black':'black','White':'white','white':'white','navyblue':'blue',

             'lightblue':'blue','blue':'blue','skyblue':'blue','darkblue':'blue','navy':'blue','winered':'red',

             'red':'red','rosered':'red','rose':'red','orange-red':'red','lightpink':'pink','pink':'pink',

              'armygreen':'green','green':'green','khaki':'green','lightgreen':'green','fluorescentgreen':'green',

             'gray':'grey','grey':'grey','brown':'brown','coffee':'brown','yellow':'yellow','purple':'purple',

             'orange':'orange','beige':'beige'}



df['product_color'] = df['product_color'].map(Color_map)

df['product_color']=df['product_color'].fillna('Unknown')



NominalVariables = [x for x in CategoricalVariables if x not in OrdinalVariables]

Lvl = df[NominalVariables].nunique()



ToDrop=['title','title_orig','currency_buyer', 'theme', 'crawl_month', 'tags', 'merchant_title','merchant_name',

              'merchant_info_subtitle','merchant_id','product_url','product_picture','product_id']

df = df.drop(ToDrop, axis = 1)

FinalNominalVariables = [x for x in NominalVariables if x not in ToDrop]



df_dummy = pd.get_dummies(df[FinalNominalVariables], columns=FinalNominalVariables)



df_clean = df.drop(FinalNominalVariables, axis = 1)

df_clean = pd.concat([df_clean, df_dummy], axis=1)



NumericVariablesNoTarget = [x for x in NumericVariables if x not in ['units_sold']]

df_scale=df_clean

df_scale = minmax_scaling(df_clean, columns=df_clean.columns)



print("The number of categorical variables: " + str(len(FinalNominalVariables)+len(OrdinalVariables)) +"; where 1 ordinal variable and 35 dummy variables")

print("The number of numeric variables: " + str(len(NumericVariables)))

df_scale.describe()
pca = PCA().fit(df_scale)



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), dpi=80, facecolor='w', edgecolor='k')

ax0, ax1 = axes.flatten()



ax0.plot(np.cumsum(pca.explained_variance_ratio_))

ax0.set_xlabel('Number of components')

ax0.set_ylabel('Cumulative explained variance');



ax1.bar(range(59),pca.explained_variance_)

ax1.set_xlabel('Number of components')

ax1.set_ylabel('Explained variance');



plt.show()
n_PCA_50 = np.size(np.cumsum(pca.explained_variance_ratio_)>0.5) - np.count_nonzero(np.cumsum(pca.explained_variance_ratio_)>0.5)

n_PCA_80 = np.size(np.cumsum(pca.explained_variance_ratio_)>0.8) - np.count_nonzero(np.cumsum(pca.explained_variance_ratio_)>0.8)

n_PCA_90 = np.size(np.cumsum(pca.explained_variance_ratio_)>0.9) - np.count_nonzero(np.cumsum(pca.explained_variance_ratio_)>0.9)

print("Already: " + format(n_PCA_50) + " Cover 50% of variance.")

print("Already: " + format(n_PCA_80) + " Cover 80% of variance.")

print("Already: " + format(n_PCA_90) + " Cover 90% of variance.")
pca = PCA(12).fit(df_scale)



X_pca=pca.transform(df_scale) 



plt.matshow(pca.components_,cmap='viridis')

plt.yticks([0,1,2,3,4,5,6,7,8,9,10,11,12],['1st Comp','2nd Comp','3rd Comp','4th Comp','5th Comp','6th Comp','7th Comp','8th Comp','9th Comp','10th Comp','11th Comp','12th Comp'],fontsize=10)

plt.colorbar()

plt.xticks(range(len(df_scale.columns)),rotation=0)

plt.tight_layout()

plt.show()
CompOne = pd.DataFrame(list(zip(df_scale.columns,pca.components_[0])),columns=('Name','Contribution to Component 1'),index=range(1,60,1))

CompOne = CompOne[(CompOne['Contribution to Component 1']>0.05) | (CompOne['Contribution to Component 1']< -0.05)]

CompOne
def ExtractColumn(lst,j): 

    return [item[j] for item in lst] 



PCA_vars = [0]*len(df_scale.columns)



for i, feature in zip(range(len(df_scale.columns)),df_scale.columns):

    x = ExtractColumn(pca.components_,i)

    if ((max(x) > 0.1) | (min(x) < -0.1)):

        if abs(max(x)) > abs(min(x)):

            PCA_vars[i] = max(x)

        else:

            PCA_vars[i] = min(x)                 

    else:

        PCA_vars[i] = 0



PCA_vars = pd.DataFrame(list(zip(df_scale.columns,PCA_vars)),columns=('Name','Max absolute contribution'),index=range(1,60,1))      

PCA_vars = PCA_vars[(PCA_vars['Max absolute contribution']!=0)]

PCA_vars
SPCA = SparsePCA(n_components=12)

SPCA_fit = SPCA.fit(df_scale)



plt.matshow(SPCA_fit.components_,cmap='viridis')

plt.yticks([0,1,2,3,4,5,6,7,8,9,10,11,12],['1st Comp','2nd Comp','3rd Comp','4th Comp','5th Comp','6th Comp','7th Comp','8th Comp','9th Comp','10th Comp','11th Comp','12th Comp'],fontsize=10)

plt.colorbar()

plt.xticks(range(len(df_scale.columns)),rotation=0)

plt.tight_layout()

plt.show()
SPCA_vars = [0]*len(df_scale.columns)



for i, feature in zip(range(len(df_scale.columns)),df_scale.columns):

    x = ExtractColumn(SPCA_fit.components_,i)

    if ((max(x) > 0.1) | (min(x) < -0.1)):

        if abs(max(x)) > abs(min(x)):

            SPCA_vars[i] = max(x)

        else:

            SPCA_vars[i] = min(x)                 

    else:

        SPCA_vars[i] = 0



SPCA_vars = pd.DataFrame(list(zip(df_scale.columns,SPCA_vars)),columns=('Name','Max absolute contribution'),index=range(1,60,1))      

SPCA_vars = SPCA_vars[(SPCA_vars['Max absolute contribution']!=0)]

SPCA_vars
KPCA = KernelPCA(n_components = len(df_scale.columns), kernel="rbf", fit_inverse_transform=True, gamma=10)

KPCA_fit = KPCA.fit(df_scale)

X_KPCA = KPCA.fit_transform(df_scale)

X_KPCA_back = KPCA.inverse_transform(X_KPCA)
NNMF = NMF(n_components=12)

NMF_fit = NNMF.fit(df_scale)



GMax = 0

for i in range(len(NMF_fit.components_)):

    Lmax = max(NMF_fit.components_[i])

    if Lmax > GMax:

        GMax = Lmax

    else:

        GMax = GMax

        

ScaledList = NMF_fit.components_ / GMax



plt.matshow(ScaledList,cmap='viridis')

plt.yticks([0,1,2,3,4,5,6,7,8,9,10,11,12],['1st Comp','2nd Comp','3rd Comp','4th Comp','5th Comp','6th Comp','7th Comp','8th Comp','9th Comp','10th Comp','11th Comp','12th Comp'],fontsize=10)

plt.colorbar()

plt.xticks(range(len(df_scale.columns)),rotation=0)

plt.tight_layout()

plt.show()
NMF_vars = [0]*len(df_scale.columns)



for i, feature in zip(range(len(df_scale.columns)),df_scale.columns):

    x = ExtractColumn(ScaledList,i)

    if ((max(x) > 0.1) | (min(x) < -0.1)):

        if abs(max(x)) > abs(min(x)):

            NMF_vars[i] = max(x)

        else:

            NMF_vars[i] = min(x)                 

    else:

        NMF_vars[i] = 0



NMF_vars = pd.DataFrame(list(zip(df_scale.columns,NMF_vars)),columns=('Name','Max absolute contribution'),index=range(1,60,1))      

NMF_vars = NMF_vars[(NMF_vars['Max absolute contribution']!=0)]

NMF_vars
All_Features = np.unique(list(PCA_vars['Name'])+list(SPCA_vars['Name'])+list(NMF_vars['Name']))



All_Features_df =  pd.DataFrame(zip(All_Features,[False]*len(All_Features),[False]*len(All_Features),

                                [False]*len(All_Features)),columns=['Feature','Is in PCA','Is in SPCA','Is in NMF'])



All_Features_df['Is in PCA'] = [True if x in list(PCA_vars['Name']) else False for x in All_Features]

All_Features_df['Is in SPCA'] = [True if x in list(SPCA_vars['Name']) else False for x in All_Features]

All_Features_df['Is in NMF'] = [True if x in list(NMF_vars['Name']) else False for x in All_Features]



All_Features_df=All_Features_df.sort_values('Feature')



All_Features_df
print(format(sum(All_Features_df['Is in PCA'])) + " features by PCA; " + format(sum(All_Features_df['Is in SPCA'])) + " features by SPCA; " +

     format(sum(All_Features_df['Is in NMF'])) + " features by NMF. ")
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')

ax0, ax1, ax2 = axes.flatten()



ScaledListPCA = abs(pca.components_)

ScaledListSPCA = abs(SPCA_fit.components_)



ax0.matshow(ScaledListPCA,cmap='viridis')

ax1.matshow(ScaledListSPCA,cmap='viridis')

ax2.matshow(ScaledList,cmap='viridis')



plt.show()