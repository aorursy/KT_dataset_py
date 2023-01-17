import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split



import tensorflow as tf
data = pd.read_csv('../input/summer-products-and-sales-in-ecommerce-wish/summer-products-with-rating-and-performance_2020-08.csv')
data
columns_to_drop = ['title', 'title_orig', 'currency_buyer', 'shipping_option_name', 'urgency_text', 'merchant_title', 'merchant_name','merchant_info_subtitle',

                   'merchant_id', 'merchant_profile_picture', 'product_url', 'product_picture', 'product_id', 'tags', 'has_urgency_banner', 'theme', 'crawl_month', 'origin_country']
data.drop(columns_to_drop, axis=1, inplace=True)
data
data.isnull().sum()
data['product_variation_size_id'].value_counts()
size_ordering = ['XXS', 'XS', 'S', 'M', 'L', 'XL', 'XXL']
def ordinal_encode(data, column, ordering):

    return data[column].apply(lambda x: ordering.index(x) if x in ordering else None)
data['product_variation_size_id'] = ordinal_encode(data, 'product_variation_size_id', size_ordering)
data['product_variation_size_id']
data['product_color'].unique()
pd.get_dummies(data['product_color'])
def onehot_encode(data, column):

    dummies = pd.get_dummies(data[column])

    data = pd.concat([data, dummies], axis=1)

    data.drop(column, axis=1, inplace=True)

    return data
data = onehot_encode(data, 'product_color')
(data.dtypes == 'object').sum()
data.isnull().sum()
null_columns = ['rating_five_count', 'rating_four_count', 'rating_three_count',

                'rating_two_count', 'rating_one_count', 'product_variation_size_id']
for column in null_columns:

    data[column] = data[column].fillna(data[column].mean())
data.isnull().sum().sum()
data
y = data['units_sold']

X = data.drop('units_sold', axis=1)
scaler = MinMaxScaler()

X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
y.unique()
encoder = LabelEncoder()



y = encoder.fit_transform(y)

y_mappings = {index: label for index, label in enumerate(encoder.classes_)}

y_mappings
y
X
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
inputs = tf.keras.Input(shape=(124,))

x = tf.keras.layers.Dense(16, activation='relu')(inputs)

x = tf.keras.layers.Dense(16, activation='relu')(x)

outputs = tf.keras.layers.Dense(15, activation='softmax')(x)



model = tf.keras.Model(inputs=inputs, outputs=outputs)





model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']

)





batch_size = 32

epochs = 160



history = model.fit(

    X_train,

    y_train,

    validation_split=0.2,

    batch_size=batch_size,

    epochs=epochs,

    verbose=0

)
plt.figure(figsize=(14, 10))



epochs_range = range(1, epochs + 1)

train_loss = history.history['loss']

val_loss = history.history['val_loss']



plt.plot(epochs_range, train_loss, label="Training Loss")

plt.plot(epochs_range, val_loss, label="Validation Loss")



plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend('upper right')



plt.show()
np.argmin(val_loss)
model.evaluate(X_test, y_test)