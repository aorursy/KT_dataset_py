%pylab inline
import pandas as pd
import scipy.sparse as sparse
import seaborn as sns
from tqdm import tqdm_notebook

data = pd.read_csv("../input/BlackFriday.csv")
data.columns
USER_ID = 'User_ID'
PRODUCT_ID = 'Product_ID'
GENDER = 'Gender', 
AGE = 'Age'
OCCUPATION = 'Occupation'
CITY_CATEGORY = 'City_Category'
STAY_IN_CURRENT = 'Stay_In_Current_City_Years'
MARITAL_STATUS = 'Marital_Status'
PRODUCT_CAT_1 = 'Product_Category_1'
PRODUCT_CAT_2 = 'Product_Category_2'
PRODUCT_CAT_3 = 'Product_Category_3'
PURCHASE = 'Purchase'
sns.distplot(data[PURCHASE]);
top_product_cats = data[PRODUCT_CAT_1].value_counts()[:5].index

ax = plt.figure().add_subplot(111)
product_cat = top_product_cats[0]
sns.distplot(data[data[PRODUCT_CAT_1] == product_cat][PURCHASE], ax=ax, label="Product Category 1 = " + str(product_cat))

product_cat = top_product_cats[1]
sns.distplot(data[data[PRODUCT_CAT_1] == product_cat][PURCHASE], ax=ax, label="Product Category 1 = " + str(product_cat))

product_cat = top_product_cats[2]
sns.distplot(data[data[PRODUCT_CAT_1] == product_cat][PURCHASE], ax=ax, label="Product Category 1 = " + str(product_cat))

product_cat = top_product_cats[3]
sns.distplot(data[data[PRODUCT_CAT_1] == product_cat][PURCHASE], ax=ax, label="Product Category 1 = " + str(product_cat))

product_cat = top_product_cats[4]
sns.distplot(data[data[PRODUCT_CAT_1] == product_cat][PURCHASE], ax=ax, label="Product Category 1 = " + str(product_cat))


plt.legend()
user_id_col = USER_ID
item_id_col = PRODUCT_ID
ratings = data[[user_id_col, item_id_col]]

num_users = ratings[user_id_col].nunique()
num_items = ratings[item_id_col].nunique()
possible_combinations = num_users * num_items
nnz = len(ratings)
nnz_percent = nnz / possible_combinations

print("Num Users:", num_users)
print("Num Items:", num_items)
print("Sparsity:", nnz_percent)
print("Not very sparse. CF will work wonders here.")

# average number of hotel_clusters per user
item_per_user = ratings.groupby(user_id_col)[item_id_col].nunique()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(211)
display(item_per_user.describe().to_frame("Number of items per user"))
sns.distplot(item_per_user, kde=False, ax=ax)
ax.set_title("Median number of items per user: {:.2f}".format(item_per_user.median()))

# # average number of users per hotel
user_per_item = ratings.groupby(item_id_col)[user_id_col].nunique()
ax = fig.add_subplot(212)
display(user_per_item.describe().to_frame("Number of users per item"))
sns.distplot(user_per_item, kde=False, ax=ax)
ax.set_title("Median number of users per item: {:.2f}".format(user_per_item.median()))

fig.tight_layout()
# Create mappings
mid_to_idx = {}
idx_to_mid = {}
for (idx, mid) in enumerate(data[item_id_col].unique().tolist()):
    mid_to_idx[mid] = idx
    idx_to_mid[idx] = mid
    
uid_to_idx = {}
idx_to_uid = {}
for (idx, uid) in enumerate(data[user_id_col].unique().tolist()):
    uid_to_idx[uid] = idx
    idx_to_uid[idx] = uid
    
def map_ids(row, mapper):
    return mapper[row]


I = data[user_id_col].apply(map_ids, args=[uid_to_idx]).values
J = data[item_id_col].apply(map_ids, args=[mid_to_idx]).values
V = np.ones(I.shape[0])
purchases = sparse.coo_matrix((V, (I, J)), dtype=np.float64)
purchases = purchases.tocsr()
def train_test_split(ratings, min_count, item_fraction, user_fraction=None):
    """
    Split recommendation data into train and test sets
    
    Params
    ------
    ratings : scipy.sparse matrix
        Interactions between users and items.
    min_count : int
        Number of user-item-interactions per user to be considered
        a part of the test set
    item_fraction : float
        Fraction of users' items to go to the test set
    user_fraction : float
        Fraction of users to split off some of their
        interactions into test set. If None, then all 
        users are considered.
    """
    # Note: likely not the fastest way to do things below.
    train = ratings.copy().tocoo()
    test = sparse.lil_matrix(train.shape)
    
    if user_fraction:
        try:
            user_index = np.random.choice(
                np.where(np.bincount(train.row) >= min_count)[0], 
                replace=False,
                size=np.int32(np.floor(user_fraction * train.shape[0]))
            ).tolist()
        except:
            print(('Not enough users with > {} '
                  'interactions for fraction of {}')\
                  .format(min_count, user_fraction))
            raise
    else:
        user_index = range(train.shape[0])
        
    train = train.tolil()

    for user in user_index:
        test_ratings = np.random.choice(ratings.getrow(user).indices, 
                                        size=int(len(ratings.getrow(user).indices) * item_fraction), 
                                        replace=False)
        train[user, test_ratings] = 0.
        # These are just 1.0 right now
        test[user, test_ratings] = ratings[user, test_ratings]
   
    # Test and training are truly disjoint
    assert(train.multiply(test).nnz == 0)
    return train.tocsr(), test.tocsr(), user_index
# 25% of the users with minimum 53 purchases will have 25% of their ratings be part of the test data
# 53 is the median 
train, test, user_index = train_test_split(purchases, 53, 0.25, user_fraction=0.25)
train
test
# top k
k = 5
# random recommendations
random_recommendations = np.random.randint(0, test.shape[1], size=(test.shape[0], k))

# most popular k items
np_num_purchases_per_item = train.sum(axis=0)
top_recommendations = np_num_purchases_per_item.argsort()[::-1][:k]
def apk(y, recommendations, k):
    precs = []
    for i in range(y.shape[0]):
        y_i = y[i][0]
        # skip empty
        if len(y_i.nonzero()[1]) == 0:
            continue
        yhat_i = recommendations[i]
        hits = len(set(recommendations[7]).intersection(set(y_i.nonzero()[1])))
        precs.append(hits / k)

    return np.mean(precs)

def coverage_at_k(y, recommendations, k):
    all_recommended_items = set(recommendations[:, :k].ravel())
    coverage = len(all_recommended_items) / y.shape[1]
    return coverage
# top k
k = 5

# bootstrap evaluation
n_repeats = 10
list_random_apk = []
list_top_apk = []
list_random_covk = []
list_top_covk = []

for i in range(n_repeats):
    train, test, user_index = train_test_split(purchases, 53, 0.25, user_fraction=0.25)
    
    # random recommendations
    random_recommendations = np.random.randint(0, test.shape[1], size=(test.shape[0], k))

    # most popular k items
    np_num_purchases_per_item = train.sum(axis=0)
    top_recommendations = np_num_purchases_per_item.argsort()[::-1][:, :k]
    top_recos_mat = np.tile(np.asarray(top_recommendations), test.shape[0]).reshape((test.shape[0], k))
    
    random_reco_apk5 = apk(test, random_recommendations, 5)
    top_reco_apk5 = apk(test, top_recos_mat, 5)
    
    random_reco_cov5 = coverage_at_k(test, random_recommendations, 5)
    top_reco_cov_5 = coverage_at_k(test, top_recos_mat, 5)
    
    list_random_apk.append(random_reco_apk5)
    list_top_apk.append(top_reco_apk5)
    list_random_covk.append(random_reco_apk5)
    list_top_covk.append(top_reco_cov_5)
def compile_result(list_result, algorithm_name, k, column_name):
    df_result = pd.DataFrame(list_result, columns=[column_name])
    df_result["Algorithm"] = algorithm_name
    df_result["k"] = k
    return df_result

df_result1 = compile_result(list_random_apk, "Random", k, "AP")
df_result2 = compile_result(list_top_apk, "Popular", k, "AP")
df_baseline_results = df_result1.append(df_result2)

df_result1_cov = compile_result(list_random_covk, "Random", k, "Coverage")
df_result2_cov = compile_result(list_top_covk, "Popular", k, "Coverage")
df_baseline_results_cov = df_result1_cov.append(df_result2_cov)

df_baseline_results["Coverage"] = df_baseline_results_cov["Coverage"]

df_baseline_results.groupby(["Algorithm", "k"])[["AP", "Coverage"]].agg(["mean", "std"])
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer

# there's only three categories. We could do with much more...
def create_cbf_recommendations(data, k=5):
    feature_cols = ["Product_Category_1", "Product_Category_2", "Product_Category_3", ]
    data_features = data.drop_duplicates("Product_ID").set_index("Product_ID")[feature_cols]

    # impute features with the mode
    imputer = SimpleImputer(strategy='most_frequent')

    data_features_imputed = imputer.fit_transform(data_features)
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm="brute", metric="cosine").fit(data_features_imputed)

    # we average out the item feature vectors of everything a customer bought and then query from our data structure
    data_imputed_features = pd.DataFrame(imputer.transform(data[feature_cols]), columns=feature_cols, 
                                         index=data["User_ID"])
    user_bought_item_features = data_imputed_features.groupby(level=0)[feature_cols].mean()
    distances, indices = nbrs.kneighbors(user_bought_item_features)

    # and here's how you get the items
    nearest_cols = ["Item" + str(v) for v in range(1, k+1)]
    distance_cols = ["Distance" + str(v) for v in range(1, k+1)]
    return pd.DataFrame(np.hstack((indices[:, 1:k+1], distances[:, 1:k+1])), columns=nearest_cols + distance_cols)
df_recos = create_cbf_recommendations(data)
df_recos[:5]
def convert_sparse_to_df(sp_data, idx_to_uid, idx_to_mid):
    # convert sparse matrix to pandas so it can be input to our CBF function
    # (not the best way to do it, but the data is small enough anyway)
    train_coo = sp_data.tocoo()
    df_train = pd.DataFrame({"User_ID" : train_coo.row, "Product_ID" : train_coo.col})
    # map these indices to their ids
    df_train["User_ID"] = [idx_to_uid[v] for v in df_train["User_ID"]]
    df_train["Product_ID"] = [idx_to_mid[v] for v in df_train["Product_ID"]]
    
    return df_train
# top k
k = 5

# bootstrap evaluation
n_repeats = 10
list_cbf_apk = []
list_cbf_covk = []

for i in range(n_repeats):
    train, test, user_index = train_test_split(purchases, 53, 0.25, user_fraction=0.25)
    
    df_train = convert_sparse_to_df(train, idx_to_uid, idx_to_mid)
    df_train = df_train.merge(data, how='left')
    cbf_recos = create_cbf_recommendations(df_train)
    
    # get the matrix form so apk would work
    cbf_np_recos = cbf_recos.filter(like="Item").values
    
    cbf_reco_apk5 = apk(test, cbf_np_recos, 5)
    cbf_reco_cov5 = coverage_at_k(test, cbf_np_recos, 5)
    
    list_cbf_apk.append(cbf_reco_apk5)
    list_cbf_covk.append(cbf_reco_cov5)
df_result_cbf_ap = compile_result(list_cbf_apk, "CBF", k, "AP")
df_result_cbf_cov = compile_result(list_cbf_covk, "CBF", k, "Coverage")

df_result_cbf_ap["Coverage"] = df_result_cbf_cov["Coverage"]
df_result_cbf_ap.groupby(["Algorithm", "k"])[["AP", "Coverage"]].agg(["mean", "std"])
import implicit

# top k
k = 5

# bootstrap evaluation
n_repeats = 10
list_wmf_apk = []
list_wmf_covk = []

for i in range(n_repeats):
    train, test, user_index = train_test_split(purchases, 53, 0.25, user_fraction=0.25)
    # train model then create recommendations
    model = implicit.als.AlternatingLeastSquares(factors=50)
    model.fit(train.T)
    
    list_wmf_recommendation = []
    # create recommendations for all users
    for user_id in range(len(idx_to_uid)):
        recommendations = model.recommend(user_id, train, N=k)
        wmf_recommendation = [v[0] for v in recommendations]
        list_wmf_recommendation.append(wmf_recommendation)
    
    wmf_recommendation = np.array(list_wmf_recommendation)
    wmf_reco_apk5 = apk(test, wmf_recommendation, k)
    wmf_reco_cov5 = coverage_at_k(test, wmf_recommendation, k)
    
    list_wmf_apk.append(wmf_reco_apk5)
    list_wmf_covk.append(wmf_reco_cov5)
df_result_wmf_ap = compile_result(list_wmf_apk, "WMF", k, "AP")
df_result_wmf_cov = compile_result(list_wmf_covk, "WMF", k, "Coverage")

df_result_wmf_ap["Coverage"] = df_result_wmf_cov["Coverage"]
df_result_wmf_ap.groupby(["Algorithm", "k"])[["AP", "Coverage"]].agg(["mean", "std"])
user_features_cols = ["Gender", "Age", "Occupation", "City_Category", "Stay_In_Current_City_Years", "Marital_Status"]
user_features = data.drop_duplicates("User_ID").set_index("User_ID")[user_features_cols]

# generate new feature - average of what users purchase in their basket
mean_purchases_per_user = data.groupby("User_ID")["Purchase"].mean()

# numpy array
np_user_features = user_features.join(mean_purchases_per_user).values
# transform occupation and city to dummies form
user_features_dummies = pd.get_dummies(user_features[["Occupation", "City_Category"]])

# transform gender to binary
user_gender = pd.Series(np.where(user_features["Gender"] == "F", 1, 0), index=user_features.index, name="Gender")

# transform age to ordinal
age_to_value = {"0-17" : 0, "18-25" : 1,"26-35" : 2, "36-45" : 3, "46-50" : 4, "51-55" : 5, "55+" : 6}
age_feature = user_features["Age"].replace(age_to_value)

# transform stay to ordinal
stay_in_city = pd.to_numeric(user_features["Stay_In_Current_City_Years"].str.replace("+", ""))

user_features_processed = user_features_dummies.join(user_gender)\
                                                .join(age_feature)\
                                                .join(stay_in_city)\
                                                .join(user_features["Marital_Status"])

np_user_features_processed = user_features_processed.values

# concatenate to identity matrix
np_user_features = np.hstack((np.eye(len(user_features_dummies)), np_user_features_processed))
sp_user_features = sparse.csr_matrix(np_user_features)
from sklearn.impute import SimpleImputer
item_feature_cols = ["Product_Category_1", "Product_Category_2", "Product_Category_3", ]
item_features = data.drop_duplicates("Product_ID").set_index("Product_ID")[item_feature_cols]

# impute features with the mode
imputer = SimpleImputer(strategy='most_frequent')

item_features_imputed = imputer.fit_transform(item_features)
np_item_features = np.hstack((np.eye(len(item_features)), item_features_imputed))
sp_item_features = sparse.csr_matrix(np_item_features)
from lightfm import LightFM

def get_lightfm_top_k_recos(model, train, sp_user_features = None, sp_item_features = None, k =5):
    users_coo = np.repeat(range(train.shape[0]), train.shape[1])
    items_coo = np.tile(range(train.shape[1]), train.shape[0])
    
    if sp_user_features is not None:
        recommendations = model.predict(users_coo,
                                    items_coo,
                                    user_features=sp_user_features.T,
                                    item_features=sp_item_features.T)
    else:
        recommendations = model.predict(users_coo,items_coo,)
        
    recommendations = recommendations.reshape(train.shape[0],-1)
    
    # sort top . Since we're getting only the top k, the training set might reduce the recommendations 
    # so we have a multiplier
    recommendations = np.argsort(-recommendations)

    # remove training instances (already liked items)
    list_pairwise_reco = []
    for idx in range(train.shape[0]):
        list_pairwise_reco.append(recommendations[idx][~np.isin(recommendations[idx], train[idx].indices)][:k].tolist())

    return np.array(list_pairwise_reco)
NUM_EPOCHS = 20
random_seed = 1234

# top k
k = 5

# bootstrap evaluation
n_repeats = 10
list_bpr_apk = []
list_bpr_covk = []
list_bpr_if_apk = []
list_bpr_if_covk = []
list_warp_apk = []
list_warp_covk = []
list_warp_if_apk = []
list_warp_if_covk = []
list_warp_kos_apk = []
list_warp_kos_covk = []
list_warp_kos_if_apk = []
list_warp_kos_if_covk = []

for i in tqdm_notebook(range(n_repeats)):
    train, test, user_index = train_test_split(purchases, 53, 0.25, user_fraction=0.25)
    # train model
    bpr_model = LightFM(no_components=30, loss='bpr', random_state=random_seed)
    bpr_model_if = LightFM(no_components=30, loss='bpr', random_state=random_seed)
    warp_model = LightFM(no_components=30, loss='warp', random_state=random_seed)
    warp_model_if = LightFM(no_components=30, loss='warp', random_state=random_seed)
    kos_model = LightFM(no_components=30, loss='warp-kos', random_state=random_seed)
    kos_model_if = LightFM(no_components=30, loss='warp-kos', random_state=random_seed)
    
    for _ in range(NUM_EPOCHS):
        bpr_model.fit(train)
        bpr_model_if.fit(train, sp_user_features, sp_item_features)

        warp_model.fit(train)
        warp_model_if.fit(train, sp_user_features, sp_item_features)

        kos_model.fit(train)
        kos_model_if.fit(train, sp_user_features, sp_item_features)

    bpr_model_recos= get_lightfm_top_k_recos(bpr_model, train,)
    bpr_model_if_recos = get_lightfm_top_k_recos(bpr_model_if, train, sp_user_features, sp_item_features)
    warp_model_recos  = get_lightfm_top_k_recos(warp_model, train, )
    warp_model_if_recos = get_lightfm_top_k_recos(warp_model_if, train, sp_user_features, sp_item_features)
    kos_model_recos = get_lightfm_top_k_recos(kos_model, train, )
    kos_model_if_recos= get_lightfm_top_k_recos(kos_model_if, train, sp_user_features, sp_item_features)
    
    list_bpr_apk.append(apk(test, bpr_model_recos, k))
    list_bpr_covk.append(coverage_at_k(test, bpr_model_recos, k))
    
    list_bpr_if_apk.append(apk(test, bpr_model_if_recos, k))
    list_bpr_if_covk.append(coverage_at_k(test, bpr_model_if_recos, k))
    
    list_warp_apk.append(apk(test, warp_model_recos, k))
    list_warp_covk.append(coverage_at_k(test, warp_model_recos, k))
    
    list_warp_if_apk.append(apk(test, warp_model_if_recos, k))
    list_warp_if_covk.append(coverage_at_k(test, warp_model_if_recos, k))
    
    list_warp_kos_apk.append(apk(test, kos_model_recos, k))
    list_warp_kos_covk.append(coverage_at_k(test, kos_model_recos, k))
    
    list_warp_kos_if_apk.append(apk(test, kos_model_if_recos, k))
    list_warp_kos_if_covk.append(coverage_at_k(test, kos_model_if_recos, k))
df_result_bpr_ap = compile_result(list_bpr_apk, "BPR", k, "AP")
df_result_bpr_ap["Coverage"] = compile_result(list_bpr_covk, "BPR", k, "Coverage")["Coverage"]

df_result_bpr_if_ap = compile_result(list_bpr_if_covk, "BPR-IF", k, "AP")
df_result_bpr_if_ap["Coverage"] = compile_result(list_bpr_if_covk, "BPR-IF", k, "Coverage")["Coverage"]

df_result_warp_ap = compile_result(list_warp_apk, "WARP", k, "AP")
df_result_warp_ap["Coverage"] = compile_result(list_warp_covk, "WARP", k, "Coverage")["Coverage"]

df_result_warp_if_ap = compile_result(list_warp_if_apk, "WARP-IF", k, "AP")
df_result_warp_if_ap["Coverage"] = compile_result(list_warp_if_covk, "WARP-IF", k, "Coverage")["Coverage"]

df_result_warp_kos_ap = compile_result(list_warp_kos_apk, "WARP-KOS", k, "AP")
df_result_warp_kos_ap["Coverage"] = compile_result(list_warp_kos_covk, "WARP-KOS", k, "Coverage")["Coverage"]

df_result_warp_kos_if_ap = compile_result(list_warp_kos_if_apk, "WARP-KOS-IF", k, "AP")
df_result_warp_kos_if_ap["Coverage"] = compile_result(list_warp_kos_if_covk, "WARP-KOS-IF", k, "Coverage")["Coverage"]
df_pairwise_results = pd.concat((df_result_bpr_ap,df_result_bpr_if_ap,
                              df_result_warp_ap, df_result_warp_if_ap,
                              df_result_warp_kos_ap,df_result_warp_kos_if_ap))
df_pairwise_results.groupby(["Algorithm", "k"])[["AP", "Coverage"]].agg(["mean", "std"])
cm = sns.light_palette("green", as_cmap=True)

df_all_results = pd.concat((df_baseline_results, df_result_cbf_ap, df_result_wmf_ap, df_pairwise_results))\
                            .groupby(["Algorithm", "k"])[["AP", "Coverage"]].agg(["mean", "std"])

s = df_all_results.style.background_gradient(cmap=cm)
s
NUM_EPOCHS = 1
random_seed = 1234

list_all_results = []
for i in tqdm_notebook(range(n_repeats)):
    train, test, user_index = train_test_split(purchases, 53, 0.25, user_fraction=0.25)
    # train model
    bpr_model = LightFM(no_components=30, loss='bpr', random_state=random_seed)
    warp_model = LightFM(no_components=30, loss='warp', random_state=random_seed)
    kos_model = LightFM(no_components=30, loss='warp-kos', random_state=random_seed)
    
    bpr_model.fit(train, epochs=NUM_EPOCHS, num_threads=2)
    warp_model.fit(train, epochs=NUM_EPOCHS, num_threads=2)
    kos_model.fit(train, epochs=NUM_EPOCHS, num_threads=2)
    
    # pairwise recos
    bpr_model_recos= get_lightfm_top_k_recos(bpr_model, train, k=30)
    warp_model_recos  = get_lightfm_top_k_recos(warp_model, train, k=30)
    kos_model_recos = get_lightfm_top_k_recos(kos_model, train, k=30)
    
    # cbf recos
    df_train = convert_sparse_to_df(train, idx_to_uid, idx_to_mid)
    df_train = df_train.merge(data, how='left')
    cbf_recos = create_cbf_recommendations(df_train, k=30)
    
    for k in np.linspace(5, 30, 6).astype('int'):
        # random recommendation
        random_recommendations = np.random.randint(0, test.shape[1], size=(test.shape[0], k))
        cbf_recos_k = cbf_recos.filter(like="Item").values[:,:k]
        bpr_model_recos_k = bpr_model_recos[:,:k]
        warp_model_recos_k = warp_model_recos[:,:k]
        kos_model_recos_k = kos_model_recos[:,:k]
        
        list_results = [{"Algorithm" : "Random", "k" : k, "AP": apk(test, random_recommendations, k), 
                         "Coverage": coverage_at_k(test, random_recommendations, k)},
                        {"Algorithm" : "CBF", "k" : k, "AP": apk(test, cbf_recos_k, k), 
                         "Coverage": coverage_at_k(test, cbf_recos_k, k)},
                        {"Algorithm" : "BPR", "k" : k, "AP": apk(test, bpr_model_recos_k, k), 
                         "Coverage": coverage_at_k(test, bpr_model_recos_k, k)},
                        {"Algorithm" : "WARP", "k" : k, "AP": apk(test, warp_model_recos_k, k), 
                         "Coverage": coverage_at_k(test, warp_model_recos_k, k)},
                        {"Algorithm" : "WARP-KOS", "k" : k, "AP": apk(test, kos_model_recos_k, k), 
                         "Coverage": coverage_at_k(test, kos_model_recos_k, k)},
                       ]
        
        list_all_results.extend(list_results)
df_all_results = pd.DataFrame(list_all_results)
df_results_mean = df_all_results.groupby(["Algorithm", "k"])[["AP", "Coverage"]].mean()
ax = plt.figure(figsize=(12, 7)).add_subplot(111)
df_results_mean.loc["Random"]["AP"].plot(ax=ax, label='Random', style='o-', ms=10)
df_results_mean.loc["CBF"]["AP"].plot(ax=ax, label="CBF", style='*-', ms=10)
df_results_mean.loc["BPR"]["AP"].plot(ax=ax, label="BPR", style='x--', ms=10)
df_results_mean.loc["WARP"]["AP"].plot(ax=ax, label="WARP", style='8--', ms=10)
df_results_mean.loc["WARP-KOS"]["AP"].plot(ax=ax, label="WARP-KOS", style='D--', ms=10)

plt.legend()
plt.title("Average Precision from k=5 to k=30", fontsize=14);
ax = plt.figure(figsize=(12, 7)).add_subplot(111)
df_results_mean.loc["Random"]["Coverage"].plot(ax=ax, label='Random', style='o-', ms=10)
df_results_mean.loc["CBF"]["Coverage"].plot(ax=ax, label="CBF", style='*-', ms=10)
df_results_mean.loc["BPR"]["Coverage"].plot(ax=ax, label="BPR", style='x--', ms=10)
df_results_mean.loc["WARP"]["Coverage"].plot(ax=ax, label="WARP", style='8--', ms=10)
df_results_mean.loc["WARP-KOS"]["Coverage"].plot(ax=ax, label="WARP-KOS", style='D--', ms=10)

plt.legend()
plt.title("Coverage from k=5 to k=30", fontsize=14);

