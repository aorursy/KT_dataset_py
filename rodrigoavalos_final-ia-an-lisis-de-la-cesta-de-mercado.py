### Importar librerias necesarias
import numpy as np
import pandas as pd
 
import xgboost as xgb
from sklearn import metrics, model_selection
data_path = "../input/"
orders_df = pd.read_csv(data_path + "orders.csv", usecols=["order_id","user_id","order_number"])
aisles=pd.read_csv('../input/aisles.csv')
departments=pd.read_csv('../input/departments.csv')
orders=pd.read_csv('../input/orders.csv')
orderp=pd.read_csv('../input/order_products__prior.csv')
ordert=pd.read_csv('../input/order_products__train.csv')
products=pd.read_csv('../input/products.csv')
aisles.head()
print('Total pasillos: {}'.format(aisles.shape[0]))
departments.head()
print('Total departamentos: {}'.format(departments.shape[0]))
orders.head()
print('Total pedidos: {}'.format(orders.shape[0]))
orderp.head()
print('Total pedidosP: {}'.format(orderp.shape[0]))
ordert.head()
print('Total pedidosT: {}'.format(ordert.shape[0]))
products.head()
print('Total productos: {}'.format(products.shape[0]))
# Combinanación pasillos, departamentos y productos (left joined to products)
goods = pd.merge(left=pd.merge(left=products, right=departments, how='left'), right=aisles, how='left')
# para conservar '-' y hacer que los nombres de los productos sean más "estándar"
goods.product_name = goods.product_name.str.replace(' ', '_').str.lower() 

goods.head()
import matplotlib.pyplot as plt # plotting


# información básica del grupo (departamentos)
plt.figure(figsize=(12, 5))
goods.groupby(['department']).count()['product_id'].copy()\
.sort_values(ascending=False).plot(kind='bar', 
                                   #figsize=(12, 5), 
                                   title='Departments: Product #')


# información básica del grupo (top-x aisles)
top_aisles_cnt = 15
plt.figure(figsize=(12, 5))
goods.groupby(['aisle']).count()['product_id']\
.sort_values(ascending=False)[:top_aisles_cnt].plot(kind='bar', 
                                   #figsize=(12, 5), 
                                   title='Aisles: Product #')

# Volumen de departamentos de parcelas, dividido por pasillos.
f, axarr = plt.subplots(6, 4, figsize=(12, 30))
for i,e in enumerate(departments.department.sort_values(ascending=True)):
    axarr[i//4, i%4].set_title('Dep: {}'.format(e))
    goods[goods.department==e].groupby(['aisle']).count()['product_id']\
    .sort_values(ascending=False).plot(kind='bar', ax=axarr[i//4, i%4])
f.subplots_adjust(hspace=2)
# leer el archivo de pedido anterior #
prior_df = pd.read_csv(data_path + "order_products__prior.csv")

# fusionarse con el archivo de pedidos para obtener el user_id #
prior_df = pd.merge(prior_df, orders_df, how="inner", on="order_id")

# Obtenga los productos y reordene el estado de la última compra de cada usuario.#
prior_grouped_df = prior_df.groupby("user_id")["order_number"].aggregate("max").reset_index()
prior_df_latest = pd.merge(prior_df, prior_grouped_df, how="inner", on=["user_id", "order_number"])
prior_df_latest = prior_df_latest[["user_id", "product_id", "reordered"]]
prior_df_latest.columns = ["user_id", "product_id", "reordered_latest"]

# Obtenga el recuento de cada producto y el número de pedidos por parte del cliente #
prior_df = prior_df.groupby(["user_id","product_id"])["reordered"].aggregate(["count", "sum"]).reset_index()
prior_df.columns = ["user_id", "product_id", "reordered_count", "reordered_sum"]

# fusionar el df anterior con el último df#
prior_df = pd.merge(prior_df, prior_df_latest, how="left", on=["user_id","product_id"])
prior_df.head()
orders_df.drop(["order_number"],axis=1,inplace=True)

train_df = pd.read_csv(data_path + "order_products__train.csv", usecols=["order_id"])
train_df = train_df.groupby("order_id").aggregate("count").reset_index()
test_df = pd.read_csv(data_path + "sample_submission.csv", usecols=["order_id"])
train_df = pd.merge(train_df, orders_df, how="inner", on="order_id")
test_df = pd.merge(test_df, orders_df, how="inner", on="order_id")
print(train_df.shape, test_df.shape)
train_df = pd.merge(train_df, prior_df, how="inner", on="user_id")
test_df = pd.merge(test_df, prior_df, how="inner", on="user_id")
del prior_df, prior_grouped_df, prior_df_latest
print(train_df.shape, test_df.shape)
products_df = pd.read_csv(data_path + "products.csv", usecols=["product_id", "aisle_id", "department_id"])
train_df = pd.merge(train_df, products_df, how="inner", on="product_id")
test_df = pd.merge(test_df, products_df, how="inner", on="product_id")
del products_df
print(train_df.shape, test_df.shape)
train_y_df = pd.read_csv(data_path + "order_products__train.csv", usecols=["order_id", "product_id", "reordered"])
train_y_df = pd.merge(train_y_df, orders_df, how="inner", on="order_id")
train_y_df = train_y_df[["user_id", "product_id", "reordered"]]
#print(train_y_df.reordered.sum())
train_df = pd.merge(train_df, train_y_df, how="left", on=["user_id", "product_id"])
train_df["reordered"].fillna(0, inplace=True)
print(train_df.shape)
#print(train_df.reordered.sum())
del train_y_df
# target variable for train set #
train_y = train_df.reordered.values

# marco de datos para las predicciones del conjunto de pruebas #
out_df = test_df[["order_id", "product_id"]]

# soltar las columnas innecesarias #
train_df = np.array(train_df.drop(["order_id", "user_id", "reordered"], axis=1))
test_df = np.array(test_df.drop(["order_id", "user_id"], axis=1))
print(train_df.shape, test_df.shape)
# función para ejecutar el modelo xgboost #
def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0):
        params = {}
        params["objective"] = "binary:logistic"
        params['eval_metric'] = 'logloss'
        params["eta"] = 0.05
        params["subsample"] = 0.7
        params["min_child_weight"] = 10
        params["colsample_bytree"] = 0.7
        params["max_depth"] = 8
        params["silent"] = 1
        params["seed"] = seed_val
        num_rounds = 100
        plst = list(params.items())
        xgtrain = xgb.DMatrix(train_X, label=train_y)

        if test_y is not None:
                xgtest = xgb.DMatrix(test_X, label=test_y)
                watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
                model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50, verbose_eval=10)
        else:
                xgtest = xgb.DMatrix(test_X)
                model = xgb.train(plst, xgtrain, num_rounds)

        pred_test_y = model.predict(xgtest)
        return pred_test_y
# ejecuta el modelo xgboost #
pred = runXGB(train_df, train_y, test_df)
del train_df, test_df

# Usa valor cut-off para obtener las predicciones #
cutoff = 0.2
pred[pred>=cutoff] = 1
pred[pred<cutoff] = 0
out_df["Pred"] = pred
out_df = out_df.ix[out_df["Pred"].astype('int')==1]
# cuando hay más de 1 producto, fusionarlos en una sola cadena #
def merge_products(x):
    return " ".join(list(x.astype('str')))
out_df = out_df.groupby("order_id")["product_id"].aggregate(merge_products).reset_index()
out_df.columns = ["order_id", "products"]
# lea el archivo csv de muestra y rellene los productos de las predicciones #
sub_df = pd.read_csv(data_path + "sample_submission.csv", usecols=["order_id"])
sub_df = pd.merge(sub_df, out_df, how="left", on="order_id")

# cuando no hay predicciones usa "ninguna" #
sub_df["products"].fillna("None", inplace=True)
sub_df.to_csv("xgb_starter_3450.csv", index=False)