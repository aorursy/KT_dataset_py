import pandas as pd

import tensorflow as tf

import os



BATCH_SIZE = 100000





def preprocess(csvin):

    df = pd.read_csv(csvin)  # Load dataset

    print("加载完成")

    # 清除不需要的属性

    df.drop(['Id', 'groupId', 'matchId', 'matchType'], axis=1, inplace=True)

    # 缺失值清理

    df.dropna(inplace=True)

    print("清理完成")



    # 填充天梯分

    count = 0

    s = 0

    for point in df['rankPoints']:

        if point > 0:

            count += 1

            s += point

    avg = s / count



    def fill_rankPoints(point):

        if point < 1:

            return avg

        else:

            return point



    df['rankPoints'] = df['rankPoints'].map(fill_rankPoints)



    print("填充完成")



    # 归一化

    df_norm = (df - df.min()) / (df.max() - df.min())

    return df_norm





def get_compiled_model():



    model = tf.keras.Sequential([

        tf.keras.layers.Dense(30, activation='relu'),

        tf.keras.layers.Dense(30, activation='relu'),

        tf.keras.layers.Dense(1, activation='sigmoid')

    ])



    model.compile(optimizer='adam',

                  loss='MSE',

                  metrics=['MeanSquaredError'])

    return model





def main():

    print(os.listdir('../input'))

    data = preprocess("../input/pubg-finish-placement-prediction/train_V2.csv")

    test = preprocess("../input/pubg-finish-placement-prediction/test_V2.csv")



    X = data.iloc[:, :-1]

    Y = data.iloc[:, [-1]]



    Xt = test



    dataset = tf.data.Dataset.from_tensor_slices((X.values, Y.values))



    train_dataset = dataset.batch(BATCH_SIZE)



    model = get_compiled_model()

    model.fit(train_dataset, epochs=50)



    print("开始预测")

    winPlacePrec = model.predict(Xt.values, batch_size=BATCH_SIZE, verbose=1)

    print("开始加载测试集")

    df_out = pd.read_csv("../input/pubg-finish-placement-prediction/test_V2.csv")

    print("开始转换")

    df_out['winPlacePerc'] = winPlacePrec



    submission = df_out[['Id', 'winPlacePerc']]

    print("开始写入结果")

    submission.to_csv('submission.csv', index=False)





main()
