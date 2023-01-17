# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Loading Data
products_df = pd.read_csv("../input/summer-products-and-sales-in-ecommerce-wish/summer-products-with-rating-and-performance_2020-08.csv")
products_df.head()
products_df.info()
# To check Null Value Count
products_df.isnull().any().to_frame()
# Replacing Null Values
products_df['rating_five_count'] = products_df['rating_five_count'].replace(np.nan, 0)
products_df['rating_four_count'] = products_df['rating_four_count'].replace(np.nan, 0)
products_df['rating_three_count'] = products_df['rating_three_count'].replace(np.nan, 0)
products_df['rating_two_count'] = products_df['rating_two_count'].replace(np.nan, 0)
products_df['rating_one_count'] = products_df['rating_one_count'].replace(np.nan, 0)
products_df['product_color'].value_counts().iloc[:60].to_frame()
# Replacing Null and duplicates
products_df['product_color'] = products_df['product_color'].replace('White', 'white')
products_df['product_color'] = products_df['product_color'].replace('Black', 'black')
products_df['product_color'] = products_df['product_color'].replace('coolblack', 'black')

products_df['product_color'] = products_df['product_color'].replace('navyblue', 'blue')
products_df['product_color'] = products_df['product_color'].replace('lightblue', 'blue')
products_df['product_color'] = products_df['product_color'].replace('skyblue', 'blue')
products_df['product_color'] = products_df['product_color'].replace('darkblue', 'blue')
products_df['product_color'] = products_df['product_color'].replace('navy', 'blue')
products_df['product_color'] = products_df['product_color'].replace('lakeblue', 'blue')
products_df['product_color'] = products_df['product_color'].replace('purple', 'blue')
products_df['product_color'] = products_df['product_color'].replace('navy blue', 'blue')

products_df['product_color'] = products_df['product_color'].replace('winered', 'red')
products_df['product_color'] = products_df['product_color'].replace('rosered', 'red')
products_df['product_color'] = products_df['product_color'].replace('rose', 'red')
products_df['product_color'] = products_df['product_color'].replace('orange-red', 'red')
products_df['product_color'] = products_df['product_color'].replace('burgundy', 'red')
products_df['product_color'] = products_df['product_color'].replace('lightred', 'red')
products_df['product_color'] = products_df['product_color'].replace('coralred', 'red')
products_df['product_color'] = products_df['product_color'].replace('wine', 'red')
products_df['product_color'] = products_df['product_color'].replace('watermelonred', 'red')

products_df['product_color'] = products_df['product_color'].replace('lightpink', 'pink')
products_df['product_color'] = products_df['product_color'].replace('beige', 'pink')
products_df['product_color'] = products_df['product_color'].replace('camel', 'pink')
products_df['product_color'] = products_df['product_color'].replace('apricot', 'pink')
products_df['product_color'] = products_df['product_color'].replace('Pink', 'pink')
products_df['product_color'] = products_df['product_color'].replace('dustypink', 'pink')

products_df['product_color'] = products_df['product_color'].replace('armygreen', 'green')
products_df['product_color'] = products_df['product_color'].replace('army green', 'green')
products_df['product_color'] = products_df['product_color'].replace('Army green', 'green')
products_df['product_color'] = products_df['product_color'].replace('lightgreen', 'green')
products_df['product_color'] = products_df['product_color'].replace('fluorescentgreen', 'green')
products_df['product_color'] = products_df['product_color'].replace('mintgreen', 'green')
products_df['product_color'] = products_df['product_color'].replace('khaki', 'green')
products_df['product_color'] = products_df['product_color'].replace('applegreen', 'green')

products_df['product_color'] = products_df['product_color'].replace('gray', 'grey')
products_df['product_color'] = products_df['product_color'].replace('silver', 'grey')
products_df['product_color'] = products_df['product_color'].replace('lightgray', 'grey')
products_df['product_color'] = products_df['product_color'].replace('lightgrey', 'grey')

products_df['product_color'] = products_df['product_color'].replace('lightyellow', 'yellow')

products_df['product_color'] = products_df['product_color'].replace('coffee', 'brown')

products_df['product_color'] = products_df['product_color'].replace('white & green', 'dual')
products_df['product_color'] = products_df['product_color'].replace('black & green', 'dual')
products_df['product_color'] = products_df['product_color'].replace('black & white', 'dual')
products_df['product_color'] = products_df['product_color'].replace('pink & grey', 'dual')
products_df['product_color'] = products_df['product_color'].replace('pink & white', 'dual')
products_df['product_color'] = products_df['product_color'].replace('black & blue', 'dual')
products_df['product_color'] = products_df['product_color'].replace('white & black', 'dual')
products_df['product_color'] = products_df['product_color'].replace('black & yellow', 'dual')
products_df['product_color'] = products_df['product_color'].replace('pink & blue', 'dual')
products_df['product_color'] = products_df['product_color'].replace('pink & black', 'dual')
products_df['product_color'] = products_df['product_color'].replace('blackwhite', 'dual')

products_df['product_color'] = products_df['product_color'].replace('multicolor', 'other')
products_df['product_color'] = products_df['product_color'].replace('floral', 'other')
products_df['product_color'] = products_df['product_color'].replace('whitefloral', 'other')
products_df['product_color'] = products_df['product_color'].replace('leopard', 'other')
products_df['product_color'] = products_df['product_color'].replace('camouflage', 'other')
products_df['product_color'] = products_df['product_color'].replace('rainbow', 'other')
products_df['product_color'] = products_df['product_color'].replace(np.nan, 'other')
# Visualizing Product Color Count
plt.figure(figsize=(10,5))
sns.countplot('product_color', data = products_df, order = products_df['product_color'].value_counts().iloc[:12].index)
plt.xlabel('Colors')
plt.ylabel('Count')
plt.title('Product Color Count')
plt.xticks(rotation=45)
plt.show()
products_df.isnull().any().to_frame()
pr_cl = products_df['product_variation_size_id'].value_counts()
pr_cl[pr_cl >1].to_frame()
# Replacing duplicates
products_df['product_variation_size_id'] = products_df['product_variation_size_id'].replace('S.', 'S')
products_df['product_variation_size_id'] = products_df['product_variation_size_id'].replace('Size S', 'S')
products_df['product_variation_size_id'] = products_df['product_variation_size_id'].replace('Size S.', 'S')
products_df['product_variation_size_id'] = products_df['product_variation_size_id'].replace('s', 'S')
products_df['product_variation_size_id'] = products_df['product_variation_size_id'].replace('Size-S', 'S')
products_df['product_variation_size_id'] = products_df['product_variation_size_id'].replace('size S', 'S')

products_df['product_variation_size_id'] = products_df['product_variation_size_id'].replace('XS.', 'XS')
products_df['product_variation_size_id'] = products_df['product_variation_size_id'].replace('Size-XS', 'XS')
products_df['product_variation_size_id'] = products_df['product_variation_size_id'].replace('SIZE XS', 'XS')
products_df['product_variation_size_id'] = products_df['product_variation_size_id'].replace('Size -XXS', 'XXS')
products_df['product_variation_size_id'] = products_df['product_variation_size_id'].replace('SIZE-XXS', 'XXS')

products_df['product_variation_size_id'] = products_df['product_variation_size_id'].replace('M.', 'M')
products_df['product_variation_size_id'] = products_df['product_variation_size_id'].replace('Size M', 'M')

products_df['product_variation_size_id'] = products_df['product_variation_size_id'].replace('Size4XL', 'XL')
products_df['product_variation_size_id'] = products_df['product_variation_size_id'].replace('SizeL', 'L')
products_df['product_variation_size_id'] = products_df['product_variation_size_id'].replace('5XL', 'XXXXXL')
products_df['product_variation_size_id'] = products_df['product_variation_size_id'].replace('4XL', 'XXXXL')
products_df['product_variation_size_id'] = products_df['product_variation_size_id'].replace('3XL', 'XXXL')
products_df['product_variation_size_id'] = products_df['product_variation_size_id'].replace('2XL', 'XXL')

pr_cl = products_df['product_variation_size_id'].value_counts()
pr_cl[pr_cl >1].to_frame()
def df_SId(name):
    if name == 'S' \
    or name == 'XS' \
    or name == 'XXS' \
    or name == 'XXXS' \
    or name == 'M' \
    or name == 'L' \
    or name == 'XL' \
    or name == 'XXL' \
    or name == 'XXXL' \
    or name == 'XXXXL' \
    or name == 'XXXXXL' :
        return name
    else:
        return 'OTHER'

products_df['product_variation_size_id'] = products_df['product_variation_size_id'].replace(np.nan,'OTHER')
products_df['product_variation_size_id'] = products_df['product_variation_size_id'].apply(df_SId)
    
products_df['product_variation_size_id'].value_counts().to_frame()
# Visualizing Product Color Count
fig, ax = plt.subplots(figsize=(10,5))
sns.countplot('product_variation_size_id', data = products_df, order = products_df['product_variation_size_id'].value_counts().index, ax = ax)
ax.set(xlabel='Size', ylabel='Count')
plt.show()
products_df.isnull().any().to_frame()
# Replacing Null Values
products_df['has_urgency_banner'] = products_df['has_urgency_banner'].replace(np.nan,0)
products_df = products_df.drop(['urgency_text'], axis=1)
fig, ax = plt.subplots(figsize=(10,5))

sns.countplot('has_urgency_banner', data = products_df, ax=ax)
ax.set(xlabel='Urgency Banner', ylabel='Count')
plt.show()
products_df.isnull().any().to_frame()
products_df['origin_country'].value_counts().to_frame()
# Replacing by frequency

products_df['origin_country'] = products_df['origin_country'].replace(np.nan, 'CN')
fig, ax = plt.subplots(figsize=(10,5))

sns.countplot('origin_country', data = products_df, ax=ax)
ax.set(xlabel='Country', ylabel='Count')
plt.show()
products_df.isnull().any().to_frame()
products_df['merchant_profile_picture'].describe().to_frame()
products_df['merchant_name'] = products_df['merchant_name'].replace(np.nan, 'no name')
products_df['merchant_info_subtitle'] = products_df['merchant_info_subtitle'].replace(np.nan, 'no info')
products_df['merchant_profile_picture'] = products_df['merchant_profile_picture'].replace(np.nan, 'no profile picture')
products_df.isnull().any().to_frame()
df_keywords = pd.read_csv("../input/summer-products-and-sales-in-ecommerce-wish/unique-categories.sorted-by-count.csv")

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x = 'keyword', y = 'count', data = df_keywords.iloc[:20], ax = ax)
ax.set(xlabel='Keyword', ylabel='Count')

plt.xticks(rotation=45, ha='right')
plt.show()
top_tags = df_keywords.iloc[:10]['keyword'].tolist()

def tag_pop(tags):
    lst_tags = tags.split(',')
    common_elements = np.intersect1d(top_tags, lst_tags)
    return len(common_elements) / len(top_tags)
    
products_df['tag_popularity'] = products_df['tags'].apply(tag_pop)
# As most of the customers are from CN so converting the price into CNY

products_df['price'] = products_df['price'] *7.97
products_df['retail_price'] = products_df['retail_price'] *7.97
products_df['currency_buyer'] = products_df['currency_buyer'].replace('EUR', 'CNY')
products_df['price_discount'] = products_df['retail_price'] - products_df['price']
def add_tags(tags):
    lst_tag = tags.split(',')
    return len(lst_tag)
products_df['tags_count'] = products_df['tags'].apply(add_tags)
products_df[['rating','rating_one_count','rating_two_count','rating_three_count','rating_four_count','rating_five_count','rating_count']].head(10)
# Data Normalization of ratings
def ratings_nor(rate, counts) :
    if rate == 0:
        return 0
    else :
        return rate/counts
products_df['rating_one_count'] = products_df.apply(lambda x: ratings_nor(x.rating_one_count, x.rating_count), axis=1)
products_df['rating_two_count'] = products_df.apply(lambda x: ratings_nor(x.rating_two_count, x.rating_count), axis=1)
products_df['rating_three_count'] = products_df.apply(lambda x: ratings_nor(x.rating_three_count, x.rating_count), axis=1)
products_df['rating_four_count'] = products_df.apply(lambda x: ratings_nor(x.rating_four_count, x.rating_count), axis=1)
products_df['rating_five_count'] = products_df.apply(lambda x: ratings_nor(x.rating_five_count, x.rating_count), axis=1)
products_df[['rating_count','merchant_rating_count']].head(10)
def merchant_ratings_nor(c_rating,m_rating):
    if c_rating == 0:
        return 0
    else:
        return c_rating/m_rating
products_df['merchant_rating_count'] = products_df.apply(lambda x: merchant_ratings_nor(x.rating_count, x.merchant_rating_count), axis=1)
def dis(d_price, r_price):
    if d_price==0:
        return 0
    else:
        return d_price/r_price
products_df['price_discount'] = products_df.apply(lambda x: dis(x.price_discount, x.retail_price),axis=1)
# Correlation Heat Map
fig, ax = plt.subplots(figsize=(25,25))

sns.heatmap(products_df.corr(), annot=True, ax=ax)
ax.add_patch(plt.Rectangle((0,2),31,1, fill = False, edgecolor = 'blue', lw = 3))

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=45, ha='right')
plt.show()
# Creating a pivot table for product color and origin country
df_group_c = products_df[['product_color','origin_country','units_sold']]

df_group_c = df_group_c.groupby(['origin_country','product_color'],as_index=False).mean()

grouped_pivot_c = df_group_c.pivot(index='origin_country',columns='product_color')
grouped_pivot_c
# Fill the NaN values with 0
grouped_pivot_c = grouped_pivot_c.fillna(0)
grouped_pivot_c
# Visualizing the pivot table
fig, ax = plt.subplots(figsize=(15,5))
im = ax.pcolor(grouped_pivot_c, cmap='RdBu')

#label names
row_labels = grouped_pivot_c.columns.levels[1]
col_labels = grouped_pivot_c.index

#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot_c.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot_c.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()
df_group_s = products_df[['product_variation_size_id','origin_country','units_sold']]

df_group_s = df_group_s.groupby(['origin_country','product_variation_size_id'],as_index=False).mean()

grouped_pivot_s = df_group_s.pivot(index='origin_country',columns='product_variation_size_id')
grouped_pivot_s
# Fill the NaN values with 0
grouped_pivot_s = grouped_pivot_s.fillna(0)
grouped_pivot_s
# Visualizing the pivot table
fig, ax = plt.subplots(figsize=(10,5))
im = ax.pcolor(grouped_pivot_s, cmap='RdBu')

#label names
row_labels = grouped_pivot_s.columns.levels[1]
col_labels = grouped_pivot_s.index

#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot_s.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot_s.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()
products_df = products_df.drop(['currency_buyer','title','title_orig','tags','shipping_option_name'], axis=1)
products_df = products_df.drop(['merchant_title','merchant_name', 'merchant_info_subtitle', 'merchant_id', 'merchant_profile_picture'], axis = 1)
products_df = products_df.drop(['product_url','product_picture','product_id','crawl_month','theme'], axis=1)

color_dummy = pd.get_dummies(products_df['product_color'])
products_df = pd.concat([products_df, color_dummy], axis=1)

products_df.drop("product_color", axis = 1, inplace=True)
products_df.head()
size_dummy = pd.get_dummies(products_df['product_variation_size_id'])
products_df = pd.concat([products_df, size_dummy], axis=1)

products_df.drop("product_variation_size_id", axis = 1, inplace=True)
products_df.head()
country_dummy = pd.get_dummies(products_df['origin_country'])
products_df = pd.concat([products_df, country_dummy], axis=1)

products_df.drop("origin_country", axis = 1, inplace=True)
products_df.head()
from sklearn.model_selection import train_test_split

X = products_df.loc[:,products_df.columns !='units_sold']
y = products_df['units_sold']
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=66)
from sklearn.neighbors import KNeighborsClassifier

training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 10
neighbors_settings = range(1,11)

for n_neighbors in neighbors_settings:
    # build the model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(knn.score(X_train, y_train))
    # record test set accuracy
    test_accuracy.append(knn.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("K")
plt.legend()
# Applying KNN with probably the best value of K
knn = KNeighborsClassifier(n_neighbors=9).fit(X_train, y_train)

print('Accuracy on training set {:.3f} '.format(knn.score(X_train,y_train)))
print('Accuracy on testing set {:.3f} '.format(knn.score(X_test,y_test)))
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train,y_train)

print('Accuracy on training set {:.3f} '.format(tree.score(X_train,y_train)))
print('Accuracy on testing set {:.3f} '.format(tree.score(X_test,y_test)))
tree = DecisionTreeClassifier(max_depth=5, random_state=0)
tree.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))
from sklearn import svm

for kernel in ('linear', 'poly', 'rbf'):
    SVM = svm.SVC(kernel=kernel)
    SVM.fit(X_train, y_train)
    print("Accuracy on training set for ",kernel," kernel : {:.3f}".format(SVM.score(X_train, y_train)))
    print("Accuracy on test set for ",kernel," kernel : {:.3f}".format(SVM.score(X_test, y_test)))
    print("\n")
# Choosing the best kernel for model accuracy
SVM = svm.SVC(kernel='linear')
SVM.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

# Accuracy Score of KNN
y_knn = knn.predict(X_test)
acc_knn = accuracy_score(y_test,y_knn)

# Accuracy Score of Decision Tree
y_tree = tree.predict(X_test)
acc_tree = accuracy_score(y_test,y_tree)

# Accuracy Score of SVM
y_svm = SVM.predict(X_test)
acc_svm = accuracy_score(y_test,y_svm)

# Visualizing for better comparison
Accuracies = [acc_tree,acc_knn,acc_svm]
ypos = np.arange(len(Accuracies))
fig, ax = plt.subplots(figsize=(5,5))
plt.xticks(ypos,['Desicion','KNN','SVM'])
plt.xlabel('Models')
plt.ylabel('Accuracies')
plt.title('Model Accuracies')
ax.bar(ypos,Accuracies, width=0.5)
plt.show()
