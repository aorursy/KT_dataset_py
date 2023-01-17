import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()
apple = market_train_df[market_train_df.assetCode == 'AAPL.O']
apple.shape
score_df = pd.DataFrame(apple.returnsOpenNextMktres10)
score_df['label'] = 1
score_df.label.mask(score_df.returnsOpenNextMktres10 < 0, -1.0, inplace=True)
score_df.head()
def compute_score(_score_df):
    """Args: score_df : pd.DataFrame([returnsOpenNextMktres10, label])"""
    _x_t = _score_df.iloc[:,0] * _score_df['label']
    return np.mean(_x_t) / np.std(_x_t, ddof=1)
compute_score(score_df)
def visualize(score_df):
    """plot distributions given score_df
    Args: score_df : pd.DataFrame([returnsOpenNextMktres10, label])
    """
    x_t = score_df.iloc[:,0] * score_df.iloc[:,1]
    x_t_mean, x_t_std = np.mean(x_t), np.std(x_t, ddof=1)
    real_mean, real_std = np.mean(score_df.iloc[:,0]), np.std(score_df.iloc[:,0], ddof=1)
    plt.hist(score_df.iloc[:,0].clip(-0.3,0.3), bins='auto', label="returnsOpenNextMktres10")
    plt.xlim([-0.3,0.3])
    plt.plot([real_mean, real_mean],[0,300], label="mean")
    plt.title("real returns distribution")
    plt.legend()
    plt.show()
    plt.hist(x_t.clip(-0.3,0.3), bins='auto', label='x_t')
    plt.xlim([-0.3,0.3])
    plt.plot([x_t_mean, x_t_mean],[0,300], label="mean")
    plt.legend()
    plt.title("returns distribution of predictions")
    plt.show()
visualize(score_df)
confidence = 3
score_df = pd.DataFrame(apple.returnsOpenNextMktres10)
score_df['label'] = 3
score_df.label.mask(score_df.returnsOpenNextMktres10 < 0, -3.0, inplace=True)
compute_score(score_df)
score_df.head()
visualize(score_df)
score_df = pd.DataFrame(apple.returnsOpenNextMktres10)
score_df['label'] = 1
score_df.label.mask(score_df.returnsOpenNextMktres10 < 0, -1.0, inplace=True)
x_t = score_df.iloc[:,0] * score_df.iloc[:,1]
x_t_mean, x_t_std = np.mean(x_t), np.std(x_t, ddof=1)
score_df['label'] = x_t_mean / score_df.returnsOpenNextMktres10
score_df.head()
compute_score(score_df)
visualize(score_df)
score_df = pd.DataFrame(apple.returnsOpenNextMktres10)
mean = 1
score_df['label'] = 1 / score_df.returnsOpenNextMktres10
score_df.head()
compute_score(score_df)
score_df = pd.DataFrame(apple.returnsOpenNextMktres10)
def compare(score_df):
    A,B = score_df.copy(),score_df.copy()
    A['label']=A['predictions']
    B['label']=1/(B['predictions'])
    print("[model A] score with label = target_value -> {}".format(compute_score(A)))
    print("[model B] score with label = 1 / target_value -> {}".format(compute_score(B)))
score_df['predictions'] =score_df.iloc[:,0]
compare(score_df)
score_df['predictions'] = score_df.iloc[:,0] + np.random.normal(0, 0.0001, len(score_df.iloc[:,0]))
compare(score_df)
score_df['predictions'] = score_df.iloc[:,0] + np.random.normal(0, 0.0005, len(score_df.iloc[:,0]))
compare(score_df)
score_df['predictions'] = score_df.iloc[:,0] + np.random.normal(0, 0.001, len(score_df.iloc[:,0]))
compare(score_df)
score_df['predictions'] = score_df.iloc[:,0] + np.random.normal(0, 0.01, len(score_df.iloc[:,0]))
compare(score_df)
score_df['predictions'] = score_df.iloc[:,0] + np.random.normal(0, 0.1, len(score_df.iloc[:,0]))
compare(score_df)
