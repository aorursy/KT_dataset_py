%matplotlib inline
# %load data.py

"""

    Module for dealing with all dataframe interactions

"""

from collections import namedtuple



import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split



FEATURES = namedtuple("Features", ["training", "testing"])



TARGET = namedtuple("Target", ["training", "testing"])





def load_df():

    """

        Load our dataframe

    """

    return pd.read_csv("../input/Seasons_Stats.csv")





def get_year_from_df(

    dataframe: pd.DataFrame, from_year: int, to_year: int

) -> pd.DataFrame:

    """

        Get data for specific years



        Args:

            df -> NBA dataframe with player stats

            from_year -> the year we want data from

            to_year -> the year we want up to



        Returns:

            A dataframe containing the data for the years needed

    """

    return dataframe[(dataframe["Year"] >= from_year) & (dataframe["Year"] <= to_year)]





def get_uniques_only(df: pd.DataFrame) -> pd.DataFrame:

    """

        Filter our dataframes for unique players only



        Args:

            df -> The NBA dataframe with all the players



        Returns:

            A dataframe containing only unique players

    """

    unique_years = []

    start_year, stop_year = int(df["Year"].min()), int(df["Year"].max())



    # iterate through all the years and grab only the unique player totals

    for current_year in range(start_year, stop_year + 1):

        current_df = df[df.Year == current_year]

        unique_years.append(current_df.drop_duplicates(subset="Player", keep="first"))



    return pd.concat(unique_years, ignore_index=True)





def get_nba_df(

    unique: bool = True, from_year: int = 2010, to_year: int = 2018

) -> pd.DataFrame:

    """

        Get a copy of the NBA dataframe



        Args:

            unique (default: True) -> Indicator for if we want unique players

                                      (most likely always)

            from_year (default: 2010) -> The year we want to start from

            to_year (default: 2017) -> the year we want up till



        Returns:

            A modified version of the nba dataframe

    """

    df = load_df()

    sliced_df = get_year_from_df(df, from_year, to_year)



    if unique:

        return get_uniques_only(sliced_df)



    return sliced_df





def create_data_tuple(

    feat_train: np.array,

    feat_test: np.array,

    target_train: np.array,

    target_test: np.array,

) -> tuple:

    """

        Create our models training/testing data object



        Args:

            feat_train -> training features from our dataframe

            feat_test -> testing features from our dataframe

            target_train -> training targets from our dataframe

            target_test -> testing targets from our dataframe



        Returns:

            A tuple containing our named tuples with our training

            and testing data



    """

    features = FEATURES(feat_train, feat_test)

    target = TARGET(target_train, target_test)

    return features, target





def get_train_test(feature_df: pd.DataFrame, target_df: pd.DataFrame) -> tuple:

    """

        Get the train test split up data from our dataframes



        Args:

            feature_df -> the nba player stats (features) dataframe

            target_df -> the target that were trying to obtain



        Returns:

            Returns our training and testing split data as two named tuples

    """



    # Obtain training and testing data with our test size as 30%

    feat_train, feat_test, target_train, target_test = train_test_split(

        feature_df, target_df, test_size=0.3, random_state=50

    )



    return create_data_tuple(feat_train, feat_test, target_train, target_test)





def main() -> None:

    """

        Main functionality for data.py

    """

    dataframe = load_df()

    sliced_df = get_year_from_df(dataframe, 2010, 2018)

    unique_players = get_uniques_only(sliced_df)





if __name__ == "__main__":

    """

        Tester functions for now

    """

    main()

# %load regression.py

"""

    Module for handling all linear regressions to be performed on our

    dataset

"""

import logging

from collections import namedtuple



import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import MinMaxScaler, StandardScaler







def filter_cols(dataframe: pd.DataFrame) -> tuple:

    """

        Filter unwanted columns from our nba dataframe for our linear regression model



        Args:

            df -> The nba dataframe



        Returns:

            tuple containing (nba player stats, nba player win shares)

    """

    # Columns we want to remove

    unwanted_cols = [

        "Unnamed: 0",

        "Year",

        "Player",

        "Pos",

        "Age",

        "Tm",

        "blanl",

        "blank2",

        "OWS",

        "DWS",

        "WS",

        "WS/48",

        "PER",

        "BPM",

        "OBPM",

        "DBPM",

        "eFG%",

        "TOV",

        "TS%",

        "3PAr",

        "VORP",

        "FTr",

        "ORB%",

        "DRB%",

        "TRB%",

        "AST%",

        "STL%",

        "BLK%",

        "TOV%",

        "USG%",

    ]



    # Target column we'd like

    target_col = ["WS"]



    # Grab the nba stats

    nba_stats = dataframe.drop(columns=unwanted_cols)

    nba_ws = dataframe[target_col]

    return nba_stats, nba_ws





def find_best_dimensions(dataframe: pd.DataFrame, threshold: float) -> tuple:

    """

        Find the amount of dimensions that maintains a specific amount 

        of information preserved



        Args:

            dataframe -> The nba dataframe to apply pca on

            threshold -> the amount of information we'd like to preserve



        Returns:

            tuple containing the reduced components, dimensions, and

            information preserved via PCA

    """

    dimensions = 0

    info_preserved = 0



    while info_preserved < threshold:

        dimensions += 1

        logging.debug(f"  - Reducing our model to {dimensions} dimensions")

        pca = PCA(n_components=dimensions)

        components = pca.fit_transform(dataframe)

        info_preserved = pca.explained_variance_ratio_.cumsum()[-1]



    return components, dimensions, info_preserved





def apply_pca(

    dataframe: pd.DataFrame, dimensions: int = 0, threshold: float = 0.95

) -> pd.DataFrame:

    """

        Apply pca to our nba dataframe given the dimensionality

        we tend to reduce to



        Args:

            df -> The nba dataframe

            dimensions -> The dimensions we tend to reduce to (if 0, auto detect based on our threshold)

            threshold -> The threshold for information preserved by our pca model.

                         Needed for determining the best number of dimensions



        Returns:

            PCA scaled dataframe



    """

    pca = None

    components = None

    info_preserved = 0



    if dimensions:

        pca = PCA(n_components=dimensions)

        components = pca.fit_transform(dataframe)

        # Grab the total information preserved by the dimension we're using

        info_preserved = pca.explained_variance_ratio_.cumsum()[-1] * 100

    else:

        logging.debug(

            f"  - No dimensions provided, finding a dimension that preserves {threshold * 100}% of the original information"

        )

        components, dimensions, info_preserved = find_best_dimensions(

            dataframe, threshold

        )



    logging.debug(

        f"  - PCA preserved {info_preserved * 100:.2f}% information with {dimensions} reduced dimensions"

    )

    # Construct our new pca dataframe

    pca_df = pd.DataFrame(

        data=components, columns=["pca-" + str(x + 1) for x in range(dimensions)]

    )

    return pca_df





def apply_scaling(features: pd.DataFrame, scale_type: str = "Standard") -> pd.DataFrame:

    """

        apply scaling to our dataframe



        Args:

            features -> the dataframe containing player stats/features

            scale_type -> the type of scaling that we'd like to apply our data



        Returns:

            The scaled dataframe

    """

    scaled_features = features

    if scale_type == "Standard":

        std_scaler = StandardScaler()

        std_features = std_scaler.fit_transform(features)

        scaled_features = pd.DataFrame(std_features, columns=scaled_features.columns)



    elif scale_type == "MinMax":

        mm_scaler = MinMaxScaler()

        mm_features = mm_scaler.fit_transform(features)

        scaled_features = pd.DataFrame(mm_features, columns=scaled_features.columns)



    return scaled_features





def create_linear_regression(

    features: namedtuple, target: namedtuple

) -> LinearRegression:

    """

        Create the linear regression model



        Args:

            training_data -> tuple of both training X and Y data

            testing_data -> tuple of both testing X and Y data



        Returns:

            Linear regression model

    """

    reg_model = LinearRegression()

    reg_model.fit(features.training, target.training)

    model_r2_score = reg_model.score(features.testing, target.testing)



    # Get model scores

    prediction = reg_model.predict(features.testing)

    sk_r2_score = r2_score(target.testing, prediction)

    mean_sqrd_err = mean_squared_error(target.testing, prediction)



    logging.debug(f"Model predicted r2 score: {model_r2_score}")

    logging.debug(f"Sklearn predicted r2 score: {sk_r2_score}")

    logging.debug(f"Mean squared error: {mean_sqrd_err}")

    return reg_model





# The available model types we can use for linear regression

MODELTYPES = {

    1: "standard",

    2: "stdscaled",

    3: "mmscaled",

    4: "pca",

    5: "stdpca",

    6: "mmpca",

}



# Linear regression container

# regression - The linear regression model

# stats - The dataframe containing the player stats

# ws - The dataframe containing the total win shares

# features - The training and testing features

# target - The training and testing targets

LINEARREG = namedtuple(

    "LinearRegression", ["regression", "stats", "ws", "features", "target"]

)





def obtain_linear_reg(

    model_type: int = 0,

    pca_dimensions: int = 3,

    pca_threshold: float = 0.95,

    from_year: int = 2010,

    to_year: int = 2018,

) -> LINEARREG:

    """

        Obtain a linear regression model



        Args:

            model_type -> the type of model data we'd like to build our regression with

            pca_dimensions -> the number of dimensions to apply to pca (if 0, auto-detect the dimensions)

            pca_threshold -> The threshold for information preserved by our pca models

            from_year -> the year we want our nba data to be selected from

            to_year -> the year we want our nba data up to



        Returns:

            Linear regression model using our customized nba dataset

    """

    logging.debug("----OBTAINING NEW REGRESSION MODEL----")

    nba_stats, nba_ws = filter_cols(get_nba_df(from_year=from_year, to_year=to_year))

    nba_stats = nba_stats.fillna(0)



    # The model we'd like

    scaling = MODELTYPES.get(model_type, "no scaling")



    logging.debug(f"Applying {scaling} to our data")

    # obtain correct data

    if scaling == "stdscaled":

        nba_stats = apply_scaling(nba_stats)

    elif scaling == "mmscaled":

        nba_stats = apply_scaling(nba_stats, scale_type="MinMax")

    elif scaling == "pca":

        nba_stats = apply_pca(nba_stats, pca_dimensions, pca_threshold)

    elif scaling == "stdpca":

        nba_stats = apply_pca(apply_scaling(nba_stats), pca_dimensions, pca_threshold)

    elif scaling == "mmpca":

        nba_stats = apply_pca(

            apply_scaling(nba_stats, scale_type="MinMax"), pca_dimensions, pca_threshold

        )



    # Obtain features and target data

    features, target = get_train_test(nba_stats, nba_ws)



    logging.debug(

        f"Creating linear regression model comprised of {len(nba_stats.columns)} features"

    )

    reg_model = create_linear_regression(features, target)



    logging.debug("----FINISHED OBTAINING REGRESSION MODEL----\n")



    # Return the regression model, nba player stats, and win shares

    return LINEARREG(reg_model, nba_stats, nba_ws, features, target)





def main() -> None:

    """

        Main functionality of our linear regression

    """

    # Gather the necessary features

    nba_stats, nba_ws = filter_cols(get_nba_df(from_year=2000))

    nba_pca = apply_pca(nba_stats.fillna(0), dimensions=5)

    std_nba = apply_scaling(nba_stats.fillna(0))

    mm_nba = apply_scaling(nba_stats.fillna(0), scale_type="MinMax")

    std_pca = apply_scaling(nba_pca)

    mm_pca = apply_scaling(nba_pca, scale_type="MinMax")



    # get train testing data

    features, target = get_train_test(nba_stats.fillna(0), nba_ws)

    pca_feats, pca_target = get_train_test(nba_pca, nba_ws)

    std_features, std_target = get_train_test(std_nba, nba_ws)

    mm_features, mm_target = get_train_test(mm_nba, nba_ws)

    std_pca, std_pca_target = get_train_test(std_pca, nba_ws)

    mm_pca, mm_pca_target = get_train_test(mm_pca, nba_ws)



    # Create linear regression models



    # create_linear_regression(features, target)

    # create_linear_regression(pca_feats, pca_target)

    # create_linear_regression(std_features, std_target)

    # create_linear_regression(mm_features, mm_target)

    # create_linear_regression(std_pca, std_pca_target)

    # create_linear_regression(mm_pca, mm_pca_target)

    obtain_linear_reg()

    # Find number of dimensions that preserves 95% of the information from our original model

    obtain_linear_reg(model_type=4, pca_dimensions=0, pca_threshold=0.95)





if __name__ == "__main__":

    LOG_FORMAT = "%(name)s - %(levelname)s - \t%(message)s"

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    main()

# Grab a standard linear regression with no preprocessing/scaling done

linear_reg = obtain_linear_reg()
# We load our player statistics that were used to calculate the model

linear_reg.stats
import seaborn as sns

sns.distplot(linear_reg.ws)
corr = linear_reg.stats.corr()

sns.heatmap(corr)
output_list = []



# get the mean and max value

for column in linear_reg.stats.columns:

    mean = linear_reg.stats[column].mean()

    max_val = linear_reg.stats[column].max()

    output_list.append((column, mean, max_val))



# Output the results

for data in output_list:

    print(f"{data[0]}\t|\tMean: {data[1]:.2f}\t|\tMax: {data[2]}")
from sklearn.metrics import mean_squared_error, r2_score

prediction = linear_reg.regression.predict(linear_reg.features.testing)

score = r2_score(prediction, linear_reg.target.testing)

score
mean_squared_error(prediction, linear_reg.target.testing)
testing = linear_reg.target.testing.values

print("Actual\t\t-\t Prediction")

for i in range(len(prediction[:20])):

    print(f"{prediction[i]}\t|\t{testing[i]}")

    print()
# Obtain a model type 4 (plain pca) linear regression

pca_reg = obtain_linear_reg(model_type=4, pca_dimensions=1)
prediction = pca_reg.regression.predict(pca_reg.features.testing)

score = r2_score(prediction, pca_reg.target.testing)

score
mean_squared_error(prediction, pca_reg.target.testing)
testing = pca_reg.target.testing.values

print("Actual\t\t-\t Prediction")

for i in range(len(prediction[:20])):

    print(f"{prediction[i]}\t|\t{testing[i]}")

    print()
# Plot our 1 dimension reduced pca model vs our 

sns.regplot(x=pca_reg.stats['pca-1'], y=pca_reg.ws['WS'])
# Obtain a model type 5 (standard scaled pca) and model type 6 (MinMax scaled pca) linear regressions

standard_pca_reg = obtain_linear_reg(model_type=5, pca_dimensions=1)

mm_pca_reg = obtain_linear_reg(model_type=6, pca_dimensions=1)
## Evaluation of standard model

prediction = standard_pca_reg.regression.predict(standard_pca_reg.features.testing)

score = r2_score(prediction, standard_pca_reg.target.testing)

score
mean_squared_error(prediction, standard_pca_reg.target.testing)
sns.regplot(x=standard_pca_reg.stats['pca-1'], y=standard_pca_reg.ws['WS'])
## Evaluation of standard model

prediction = mm_pca_reg.regression.predict(mm_pca_reg.features.testing)

score = r2_score(prediction, mm_pca_reg.target.testing)

score
mean_squared_error(prediction, mm_pca_reg.target.testing)
sns.regplot(x=mm_pca_reg.stats['pca-1'], y=pca_reg.ws['WS'])
# Find us a dimension that preserves the amount of information we're looking for.

thresholds = [.95, .96, .97, .98, .99]

results = []

for threshold in thresholds:

    model = obtain_linear_reg(model_type=4, pca_dimensions=0, pca_threshold=threshold)

    print(f"To preserve {threshold * 100:.2f}% information, we need: {len(model.stats.columns)} dimensions")
# Find us a dimension that preserves the amount of information we're looking for. 

thresholds = [.95, .96, .97, .98, .99]

results = []

for threshold in thresholds:

    model = obtain_linear_reg(model_type=5, pca_dimensions=0, pca_threshold=threshold)

    print(f"To preserve {threshold * 100:.2f}% information, we need: {len(model.stats.columns)} dimensions")
# Find us a dimension that preserves the amount of information we're looking for.

thresholds = [.95, .96, .97, .98, .99]

results = []

for threshold in thresholds:

    model = obtain_linear_reg(model_type=6, pca_dimensions=0, pca_threshold=threshold)

    print(f"To preserve {threshold * 100:.2f}% information, we need: {len(model.stats.columns)} dimensions")
pca_model = obtain_linear_reg(model_type=4, pca_dimensions=2)
## Evaluation of standard model

prediction = pca_model.regression.predict(pca_model.features.testing)

score = r2_score(prediction, pca_model.target.testing)

score
mean_squared_error(prediction, pca_model.target.testing)
std_pca_model = obtain_linear_reg(model_type=5, pca_dimensions=2)
## Evaluation of standard model

prediction = std_pca_model.regression.predict(std_pca_model.features.testing)

score = r2_score(prediction, std_pca_model.target.testing)

score
mean_squared_error(prediction, std_pca_model.target.testing)
std_pca_model = obtain_linear_reg(model_type=5, pca_dimensions=9)
## Evaluation of standard model

prediction = std_pca_model.regression.predict(std_pca_model.features.testing)

score = r2_score(prediction, std_pca_model.target.testing)

score
mean_squared_error(prediction, std_pca_model.target.testing)
mm_pca_model = obtain_linear_reg(model_type=6, pca_dimensions=8)
## Evaluation of standard model

prediction = mm_pca_model.regression.predict(mm_pca_model.features.testing)

score = r2_score(prediction, mm_pca_model.target.testing)

score
mean_squared_error(prediction, mm_pca_model.target.testing)