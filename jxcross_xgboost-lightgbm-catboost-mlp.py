import xgboost as xgb



model_xgb = xgb.XGBClassifier(max_depth=9, learning_rate=0.01, n_estimators=500, reg_alpah=1.1,

                             colsample_bytree=0.9, subsample=0.9, n_jobs=5)

model_xgb.fit(X_tr, y_tr, eval_set=[(X_vld, y_vld)], verbose=False, early_stopping_rounds=50)

pred_xgb = model_xgb.predict(X_vld)

score_xgb = metrics.accuracy_score(pred_xgb, y_vld)

print("XGBoost Test score: ", score_xgb)
import lightgbm as lgbm



model_lgbm = lgbm.LGBMClassifier(max_depth=9, lambda_l1=0.1, lambda_l2=0.01, learning_rate=0.01,

                               n_estimators=500, reg_alpha=1.1, colsample_bytree=0.9, subsample=0.9, n_jobs=5)

model_lgbm.fit(X_tr, y_tr, eval_set=[(X_vld, y_vld)], verbose=False, early_stopping_rounds=50,

              eval_metric="accuracy")

pred_lgbm = model_lgbm.predict(X_vld)

score_lgbm = metrics.accuracy_score(pred_lgbm, y_vld)

print("LightGBM Test Score: ", score_lgbm)
import catboost as cboost



model_cboost = cboost.CatBoostClassifier(depth=9, reg_lambda=0.1, learning_rate=0.01, iterations=500)

model_cboost.fit(X_tr, y_tr, eval_set=[(X_vld, y_vld)], verbose=False, early_stopping_rounds=50)

pred_cboost = model_cboost.predict(X_vld)

score_cboost = metrics.accuracy_score(pred_cboost, y_vld)

print("CatBoost Test Score: ", score_cboost)



import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, BatchNormalization, Activation

from keras.optimizers import Adam

from keras import backend as K



model_mlp = Sequential()

model_mlp.add(Dense(45 ,activation='linear', input_dim=13))

model_mlp.add(BatchNormalization())



model_mlp.add(Dense(9,activation='linear'))

model_mlp.add(BatchNormalization())

model_mlp.add(Dropout(0.4))



model_mlp.add(Dense(5,activation='linear'))

model_mlp.add(BatchNormalization())

model_mlp.add(Dropout(0.2))



model_mlp.add(Dense(1,activation='relu', ))



optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)

model_mlp.compile(optimizer=optimizer, 

            loss='binary_crossentropy', 

            metrics=['accuracy'])



hist = model_mlp.fit(X_tr, y_tr, epochs=500, batch_size=30, validation_data=(X_vld,y_vld), verbose=False)



pred_mlp = model_mlp.predict_classes(X_vld)[:,0]

score_mlp = metrics.accuracy_score(pred_mlp, y_vld)

print("MLP Test Score: ", score_mlp)