import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt

from xgboost import XGBClassifier

from xgboost import plot_importance

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report, mean_squared_error

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV

def load_csv(dataset, datadir):

    """Load covid19-week1 kaggle data sets"""

    df = pd.read_csv(

        f"{datadir}/{dataset}.csv",

        dtype={"Country/Region": "category"},

        parse_dates=["Date"],

    )

    df["DayOfYear"] = df["Date"].dt.dayofyear

    df["Province/State"].fillna(df["Country/Region"], inplace=True)

    return df





def sigmoid(x, x0, L, k):

    return L / (1 + np.exp(-k * (x - x0)))





def fit(df: pd.DataFrame, index: str, channel: str):

    x = df[index].values

    y = df[channel].values

    try:

        p, pc = curve_fit(

            sigmoid,

            x,

            y,

            p0=(np.median(x), y[-1], 0.1),

            maxfev=100000,

            bounds=([0, y[-1], 0], [np.inf, np.inf, 2]),

            ftol=1e-5,

        )

    except RuntimeError:

        return [-1] * 9

    return list(p) + pc[np.triu_indices_from(pc)].tolist()





def apply_fit(df: pd.DataFrame, index: str, channel: str) -> pd.DataFrame:

    """Apply fit to channel"""

    fits = df.groupby(["Country/Region", "Province/State"])[

        [index, channel]

    ].apply(lambda x: fit(x, index=index, channel=channel))

    return pd.DataFrame(

        fits.tolist(),

        columns=["x0", "L", "k", "sx0", "sx0L", "sx0k", "sL", "sLk", "sk"],

        index=fits.index,

    ).add_suffix(f"_{channel}")





def prep_pop(filename: str) -> pd.DataFrame:

    """Prepare population summary data.



    Dataset: SYB62_1_201907_Population, Surface Area and Density.csv



    Index: country

    Columns:

        - young%: percent young 0-14 years old

        - old%: percent 60+ years old

        - density: people/area

        - population: people

        - fem%: percent females of total

    """

    df = pd.read_csv(

        filename,

        engine="python",

        skiprows=901,

        index_col=0,

        names=(

            "idx",

            "country",

            "Year",

            "Series",

            "Value",

            "Footnotes",

            "Source",

        ),

        usecols=["country", "Year", "Series", "Value"],

    )

    pivot = (

        df[df["Year"] == 2019]

        .pivot(columns="Series", values="Value")

        .dropna()

        .rename(

            columns={

                "Population aged 0 to 14 years old (percentage)": "young%",

                "Population aged 60+ years old (percentage)": "old%",

                "Population density": "density",

                "Population mid-year estimates (millions)": "population",

                "Population mid-year estimates for females (millions)": "fem",

                "Population mid-year estimates for males (millions)": "male",

                "Sex ratio (males per 100 females)": "ratio",

            }

        )

    )

    pivot["fem%"] = pivot["fem"] * 100.0 / pivot["population"]

    return pivot.drop(columns=["fem", "male", "ratio"])





def prep_hh(filename: str) -> pd.DataFrame:

    """Household statistics.



    Dataset:

        population_division_UN_Houseshold_Size_and_Composition_2019.csv



    Index: country

    Columns:

        - avg_hh: average household size

        - hh%1: percentage single housholds

        - hh%2-3: percentage 2-3 person housholds

        - hh%4-5: percentage 4-5 person housholds

        - hh%6+: percentage 6+ person housholds

        - hh65+: percentage housholds with 65+ yearolds

    """

    return (

        pd.read_csv(

            filename,

            usecols=[0, 3, 4, 5, 6, 7, 8, 18],

            names=[

                "country",

                "date",

                "avg_hh",

                "hh%1",

                "hh%2-3",

                "hh%4-5",

                "hh%6+",

                "hh65+",

            ],

            na_values="..",

            header=0,

            parse_dates=["date"],

        )

        .sort_values(by=["country", "date"])

        .groupby("country")

        .tail(1)

        .drop(columns=["date"])

        .set_index("country")

    )





def predict(df):

    df["y_ConfirmedCases"] = (

        (

            df["L_ConfirmedCases"]

            / (

                1

                + np.exp(

                    -df["k_ConfirmedCases"]

                    * (df["DayOfYear"] - df["x0_ConfirmedCases"])

                )

            )

        )

        .round()

        .astype(int)

    )

    df["y_Fatalities"] = (

        (

            df["L_Fatalities"]

            / (

                1

                + np.exp(

                    -df["k_Fatalities"]

                    * (df["DayOfYear"] - df["x0_Fatalities"])

                )

            )

        )

        .round()

        .astype(int)

    )

    return df





def merge_support_main(main_df, supp_df):

    return pd.merge(

        main_df,

        supp_df,

        left_on="Country/Region",

        right_on="country",

        how="left",

    )





def augment_dataset(DATADIR="data/"):

    # Load kaggle data

    train = load_csv("train", DATADIR)

    test = load_csv("test", DATADIR)



    # Make fits

    fits_cc = apply_fit(train, index="DayOfYear", channel="ConfirmedCases")

    fits_f = apply_fit(train, index="DayOfYear", channel="Fatalities")

    fits = fits_cc.join(fits_f)



    # Insert fits

    train = pd.merge(

        train, fits.reset_index(), on=["Country/Region", "Province/State"]

    )

    test = pd.merge(

        test, fits.reset_index(), on=["Country/Region", "Province/State"]

    )



    # Augment dataset

    ## Supports

    sf = pd.read_csv(f"{DATADIR}/supporting_features.csv", index_col=0).drop(

        columns=["avg_HH"]

    )

    sf.index.name = "country"

    train = merge_support_main(train, sf)

    test = merge_support_main(test, sf)



    ## Housholds

    hh = prep_hh(

        f"{DATADIR}/population_division_UN_Houseshold_Size_and_Composition_2019.csv"

    )

    train = pd.merge(

        train, hh, left_on="Country/Region", right_on="country", how="left"

    )

    test = pd.merge(

        test, hh, left_on="Country/Region", right_on="country", how="left"

    )

    ## Population size

    pop = prep_pop(

        f"{DATADIR}/SYB62_1_201907_Population, Surface Area and Density.csv"

    )

    train = pd.merge(

        train, pop, left_on="Country/Region", right_on="country", how="left"

    )

    test = pd.merge(

        test, pop, left_on="Country/Region", right_on="country", how="left"

    )



    # Include fit predictions

    train = predict(train)

    test = predict(test)

    return train, test
## See appendix on details on augmented data

train = pd.read_csv("/kaggle/input/covid19aug/augmented/train_aug.csv")

test = pd.read_csv("/kaggle/input/covid19aug/augmented/test_aug.csv")
def plotfit(df, country, channel, ax, c="b"):

    df_sel = df[

        (df["Country/Region"] == country) & (df["Province/State"] == country)

    ]

    x = df_sel["DayOfYear"].values

    y = df_sel[channel].values

    s = df_sel[f'y_{channel}']

    (x0, L, k) = df_sel[

        [f"x0_{channel}", f"L_{channel}", f"k_{channel}"]

    ].values[0]

    s = sigmoid(x, x0, L, k)

    ax.plot(x, y, f"{c}o", mfc="none", alpha=0.5)

    ax.plot(x, s, f"{c}-")

    ax.text(

        0.2,

        0.8,

        f"$x_0$={x0:.2f}\nL={L:.2f}\nk={k:.2f}",

        transform=ax.transAxes,

        verticalalignment="top",

    )

    ax.set_title(country)

    ax.set_ylabel(channel)



sel = ("Italy", "Austria", "Germany", "Netherlands")

fig, ax = plt.subplots(4, 2, sharex=True, figsize=(10,8))

for i, country in enumerate(sel):

    plotfit(train, country, "ConfirmedCases", ax[i, 0], c="g")

    plotfit(train, country, "Fatalities", ax[i, 1], c="r")

    
def prepare_data(df: pd.DataFrame):

    df = df[df['ConfirmedCases'] > 100].reset_index()

    df['days_since_first100cases'] = df.groupby(['Province/State', 'Country/Region']

                                                )['ConfirmedCases'].transform(lambda x: range(len(x)))

    return df.drop(['x0_ConfirmedCases', 'L_ConfirmedCases', 'k_ConfirmedCases', 'sx0_ConfirmedCases',

                    'sx0L_ConfirmedCases', 'sx0k_ConfirmedCases', 'sL_ConfirmedCases',

                    'sLk_ConfirmedCases', 'sk_ConfirmedCases', 'x0_Fatalities',

                    'L_Fatalities', 'k_Fatalities', 'sx0_Fatalities', 'sx0L_Fatalities',

                    'sx0k_Fatalities', 'sL_Fatalities', 'sLk_Fatalities', 'sk_Fatalities',

                    'DayOfYear', 'index', 'Id', 'Date'], axis=1)





def impute_values(df: pd.DataFrame):

    features = ['life_expectancy_years', 'literacy_rates', 'veg_supply_person_kg_year',

                'respiratory_infections_death', 'deaths_from_smoking%', 'young%',

                'old%', 'density', 'population', 'fem%', 'number_doc_per1000',

                'surgeons_act_working', 'obstetricians_act_working',

                'anaesthesiologists_act_working', 'avg_hh', 'hh%1', 'hh%2-3', 'hh%4-5',

                'hh%6+', 'hh65+']

    df_per_country = df.groupby(

        ['Province/State', 'Country/Region'], as_index=False).min()

    df_per_country = df_per_country[[

        'Province/State', 'Country/Region']+features].replace(0, np.nan)

    # drop colunms with many nulls

    column_null = df_per_country.isnull().mean()

    column_null_drop = list(column_null[column_null > 0.5].index)

    df_per_country_clean = df_per_country.drop(

        column_null_drop, axis=1).fillna(df_per_country.median())

    return df.drop(features, axis=1).merge(df_per_country_clean,

                                           on=['Province/State', 'Country/Region'], how='left')
def model(df, ylab):

    features = df.drop(['Country/Region','Province/State',

                    'ConfirmedCases', 'Fatalities', 'Long', 'y_Fatalities', 'y_ConfirmedCases'

                    ], axis=1)

    label = df[ylab]

    train_df, test_df, train_label, test_label = train_test_split(

        features, label, test_size=0.2)

    model = XGBClassifier(objective='reg:linear')

    model.fit(train_df, train_label)

    test_pred = model.predict(test_df)

    return model, np.sqrt(mean_squared_error(test_label, test_pred))
df = impute_values(prepare_data(train))

df.head(3)
corr = df.drop(['Country/Region', 'Province/State'], axis=1).corr()

fig, ax = plt.subplots(figsize=(12,8))

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);
xgb_cases, rmse_cases= model(df,'y_ConfirmedCases')
print("ConfirmedCases: Importance by Weight")

plot_importance(xgb_cases,max_num_features=20, importance_type='gain', show_values=False)

plt.show()
xgb_fatalities, rmse_fatalities = model(df,'y_Fatalities')
print("Fatalities:  Importance by Weight")

plot_importance(xgb_fatalities, max_num_features=20, importance_type='gain', show_values=False)

plt.show()
prediction = predict(test)[["ForecastId", "y_ConfirmedCases", "y_Fatalities"]]

prediction.columns = ["ForecastId","ConfirmedCases","Fatalities"]

prediction.head()

prediction.to_csv('submission.csv', index=False)