# numerical computation
import numpy as np

# dataframes
import pandas as pd

# visualization
# we could do this with matplotlib, but I wanted to try
# something new... Do not fear: only two visualizations ;)
# altair is a very nice plotting library by the way!
import altair as alt  # plots
alt.renderers.enable("kaggle")
from IPython.display import display  # pretty display of e.g. dataframes

# progress bars
from tqdm import tqdm_notebook as tqdm

# simple models and preprocessing from scikit learn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# some constants:
num_lines = 629_145_480  # total number of lines in the CSV file
num_lines_per_segment = 150_000  # number of lines in each test segment
def create_features_for_segment(idx, raw_data):
    """ create features for a segment of raw data 
    
    Args:
        idx: index to save the features under
        raw_data: the raw data segment to calculate the features for
        
    Returns:
        features: a pandas DataFrame with 180 feature columns and a single
            index specified by `idx`.
    
    """
    data = pd.DataFrame(index=[idx])
    data.loc[idx, "data"] = raw_data.iloc[-1].item()
    data.loc[idx, "mean"] = raw_data.mean().item()
    data.loc[idx, "std"] = raw_data.std().item()
    data.loc[idx, "max"] = raw_data.max().item()
    data.loc[idx, "min"] = raw_data.min().item()
    data.loc[idx, "mad"] = raw_data.mad().item()
    data.loc[idx, "kurt"] = raw_data.kurtosis().item()
    data.loc[idx, "skew"] = raw_data.skew().item()
    data.loc[idx, "median"] = raw_data.median().item()
    data.loc[idx, "q01"] = np.quantile(raw_data, 0.01)
    data.loc[idx, "q05"] = np.quantile(raw_data, 0.05)
    data.loc[idx, "q95"] = np.quantile(raw_data, 0.95)
    data.loc[idx, "q99"] = np.quantile(raw_data, 0.99)
    data.loc[idx, "iqr"] = np.subtract(*np.percentile(raw_data, [75, 25]))
    data.loc[idx, "abs_mean"] = raw_data.abs().mean().item()
    data.loc[idx, "abs_std"] = raw_data.abs().std().item()
    data.loc[idx, "abs_max"] = raw_data.abs().max().item()
    data.loc[idx, "abs_min"] = raw_data.abs().min().item()
    data.loc[idx, "abs_mad"] = raw_data.abs().mad().item()
    data.loc[idx, "abs_kurt"] = raw_data.abs().kurtosis().item()
    data.loc[idx, "abs_skew"] = raw_data.abs().skew().item()
    data.loc[idx, "abs_median"] = raw_data.abs().median().item()
    data.loc[idx, "abs_q01"] = np.quantile(raw_data.abs(), 0.01)
    data.loc[idx, "abs_q05"] = np.quantile(raw_data.abs(), 0.05)
    data.loc[idx, "abs_q95"] = np.quantile(raw_data.abs(), 0.95)
    data.loc[idx, "abs_q99"] = np.quantile(raw_data.abs(), 0.99)
    data.loc[idx, "abs_iqr"] = np.subtract(*np.percentile(raw_data.abs(), [75, 25]))

    for window in [10, 100, 1000]:

        data_roll_mean = raw_data.rolling(window).mean().dropna()
        data.loc[idx, f"mean_mean_{window}"] = data_roll_mean.mean().item()
        data.loc[idx, f"std_mean_{window}"] = data_roll_mean.std().item()
        data.loc[idx, f"max_mean_{window}"] = data_roll_mean.max().item()
        data.loc[idx, f"min_mean_{window}"] = data_roll_mean.min().item()
        data.loc[idx, f"mad_mean_{window}"] = data_roll_mean.mad().item()
        data.loc[idx, f"kurt_mean_{window}"] = data_roll_mean.kurtosis().item()
        data.loc[idx, f"skew_mean_{window}"] = data_roll_mean.skew().item()
        data.loc[idx, f"median_mean_{window}"] = data_roll_mean.median().item()
        data.loc[idx, f"q01_mean_{window}"] = np.quantile(data_roll_mean, 0.01)
        data.loc[idx, f"q05_mean_{window}"] = np.quantile(data_roll_mean, 0.05)
        data.loc[idx, f"q95_mean_{window}"] = np.quantile(data_roll_mean, 0.95)
        data.loc[idx, f"q99_mean_{window}"] = np.quantile(data_roll_mean, 0.99)
        data.loc[idx, f"iqr_mean_{window}"] = np.subtract(
            *np.percentile(data_roll_mean, [75, 25])
        )
        data.loc[idx, f"abs_mean_mean_{window}"] = data_roll_mean.abs().mean().item()
        data.loc[idx, f"abs_std_mean_{window}"] = data_roll_mean.abs().std().item()
        data.loc[idx, f"abs_max_mean_{window}"] = data_roll_mean.abs().max().item()
        data.loc[idx, f"abs_min_mean_{window}"] = data_roll_mean.abs().min().item()
        data.loc[idx, f"abs_mad_mean_{window}"] = data_roll_mean.abs().mad().item()
        data.loc[idx, f"abs_kurt_mean_{window}"] = (
            data_roll_mean.abs().kurtosis().item()
        )
        data.loc[idx, f"abs_skew_mean_{window}"] = data_roll_mean.abs().skew().item()
        data.loc[idx, f"abs_median_mean_{window}"] = (
            data_roll_mean.abs().median().item()
        )
        data.loc[idx, f"abs_q01_mean_{window}"] = np.quantile(
            data_roll_mean.abs(), 0.01
        )
        data.loc[idx, f"abs_q05_mean_{window}"] = np.quantile(
            data_roll_mean.abs(), 0.05
        )
        data.loc[idx, f"abs_q95_mean_{window}"] = np.quantile(
            data_roll_mean.abs(), 0.95
        )
        data.loc[idx, f"abs_q99_mean_{window}"] = np.quantile(
            data_roll_mean.abs(), 0.99
        )
        data.loc[idx, f"abs_iqr_mean_{window}"] = np.subtract(
            *np.percentile(data_roll_mean.abs(), [75, 25])
        )

        data_roll_std = raw_data.rolling(window).std().dropna()
        data.loc[idx, f"mean_std_{window}"] = data_roll_std.mean().item()
        data.loc[idx, f"std_std_{window}"] = data_roll_std.std().item()
        data.loc[idx, f"max_std_{window}"] = data_roll_std.max().item()
        data.loc[idx, f"min_std_{window}"] = data_roll_std.min().item()
        data.loc[idx, f"mad_std_{window}"] = data_roll_std.mad().item()
        data.loc[idx, f"kurt_std_{window}"] = data_roll_std.kurtosis().item()
        data.loc[idx, f"skew_std_{window}"] = data_roll_std.skew().item()
        data.loc[idx, f"median_std_{window}"] = data_roll_std.median().item()
        data.loc[idx, f"q01_std_{window}"] = np.quantile(data_roll_mean, 0.01)
        data.loc[idx, f"q05_std_{window}"] = np.quantile(data_roll_mean, 0.05)
        data.loc[idx, f"q95_std_{window}"] = np.quantile(data_roll_mean, 0.95)
        data.loc[idx, f"q99_std_{window}"] = np.quantile(data_roll_mean, 0.99)
        data.loc[idx, f"iqr_std_{window}"] = np.subtract(
            *np.percentile(data_roll_std, [75, 25])
        )
        data.loc[idx, f"abs_mean_std_{window}"] = data_roll_std.abs().mean().item()
        data.loc[idx, f"abs_std_std_{window}"] = data_roll_std.abs().std().item()
        data.loc[idx, f"abs_max_std_{window}"] = data_roll_std.abs().max().item()
        data.loc[idx, f"abs_min_std_{window}"] = data_roll_std.abs().min().item()
        data.loc[idx, f"abs_mad_std_{window}"] = data_roll_std.abs().mad().item()
        data.loc[idx, f"abs_kurt_std_{window}"] = data_roll_std.abs().kurtosis().item()
        data.loc[idx, f"abs_skew_std_{window}"] = data_roll_std.abs().skew().item()
        data.loc[idx, f"abs_median_std_{window}"] = data_roll_std.abs().median().item()
        data.loc[idx, f"abs_q01_std_{window}"] = np.quantile(data_roll_std.abs(), 0.01)
        data.loc[idx, f"abs_q05_std_{window}"] = np.quantile(data_roll_std.abs(), 0.05)
        data.loc[idx, f"abs_q95_std_{window}"] = np.quantile(data_roll_std.abs(), 0.95)
        data.loc[idx, f"abs_q99_std_{window}"] = np.quantile(data_roll_std.abs(), 0.99)
        data.loc[idx, f"iqr_std_{window}"] = np.subtract(
            *np.percentile(data_roll_std, [75, 25])
        )

    return data
def load_raw_train_data():
    """ load raw train data as a dataframe """
    train_data = pd.read_csv(
        filepath_or_buffer="../input/train.csv",
        dtype={"acoustic_data": np.int16, "time_to_failure": np.float32},
    )
    return train_data
def load_train_features_and_target():
    """ load raw train data and transform to two (feature and target) dataframes """
    print("loading raw train data... [this takes about 2 min]")
    raw_data = load_raw_train_data()
    ram_mb = raw_data.memory_usage(deep=True).sum().item() / 1024 ** 2
    print(f"raw train data loaded. RAM Usage: {ram_mb:.2f} MB")
    idxs = np.arange(num_lines_per_segment, num_lines, num_lines_per_segment // 2)
    target_values = np.zeros((idxs.shape[0], 1))
    feature_values = np.zeros((idxs.shape[0], 180))
    print("transforming raw data into feature dataframe. This takes about 30 min...")
    for i, idx in enumerate(tqdm(idxs)):
        segment = raw_data.iloc[idx - num_lines_per_segment + 1 : idx + 1]
        target_values[i] = segment.time_to_failure.values[-1:]
        segment = segment[["acoustic_data"]]
        feature_row = create_features_for_segment(idx, segment)
        feature_values[i, :] = feature_row.values
    features = pd.DataFrame(
        index=idxs, data=feature_values, columns=feature_row.columns
    )
    print("train feature dataframe created")
    target = pd.DataFrame(index=idxs, data=target_values, columns=["time_to_failure"])
    print("train target dataframe created")
    return features, target
def load_test_features():
    """ load raw test data and transform to a feature dataframe """
    print("loading train segment ids...")
    seg_ids = pd.read_csv("../input/sample_submission.csv", index_col="seg_id").index
    feature_values = np.zeros((seg_ids.shape[0], 180))
    print("converting test segments into feature dataframes. "
          "This takes a about 30 min...")
    for i, seg_id in enumerate(tqdm(seg_ids)):
        segment = pd.read_csv(
            filepath_or_buffer=f"../input/test/{seg_id}.csv",
            dtype={"acoustic_data": np.int16, "time_to_failure": np.float32},
        )
        feature_row = create_features_for_segment(seg_id, segment)
        feature_values[i, :] = feature_row
    features = pd.DataFrame(
        index=seg_ids, data=feature_values, columns=feature_row.columns
    )
    print("test feature dataframe created")
    return features
# load data
features, target = load_train_features_and_target()

# scale features inplace
feature_scaler = StandardScaler(copy=False)
feature_scaler.fit_transform(features)

print("\n\n\ndata:")
print(features.shape)
display(features.head())

print("\n\n\ntarget:")
print(target.shape)
display(target.head())
model = LinearRegression()
model.fit(features, target)
def make_prediction(features, column_name="prediction"):
    """ custom prediction function that returns a dataframe in stead of a numpy array"""
    prediction = pd.DataFrame(
        index=features.index, data=model.predict(features.values), columns=[column_name]
    )
    return prediction
prediction = make_prediction(features)
np.mean(np.abs(prediction.values - target.values))
chart_data = pd.concat([prediction, target], 1).iloc[::10].reset_index()
alt.Chart(chart_data).mark_point().encode(x="prediction", y="time_to_failure")
chart1 = alt.Chart(chart_data).mark_line().encode(
    x = "index",
    y = "time_to_failure",
    
)
chart2 = alt.Chart(chart_data).mark_line().encode(
    x = "index",
    y = "prediction",
    color=alt.value("red")
)
chart1 + chart2
test_features = load_test_features()
feature_scaler.transform(test_features)
prediction = make_prediction(test_features, column_name="time_to_failure")
prediction.time_to_failure[prediction.time_to_failure < 0] = 0
prediction.index.name = "seg_id"
prediction.to_csv("submission.csv")
prediction.head()