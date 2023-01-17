from kaggle.competitions import twosigmanews
from IPython.core.display import display

env = twosigmanews.make_env()

df = env._var07.set_index(["assetCode", "time"], verify_integrity=True)
df["confidenceValue"] = 2 * (df.returnsOpenNextMktres10 > 0).astype(float) - 1

m = min(df[df.returnsOpenNextMktres10 > 0].returnsOpenNextMktres10.min(), df[df.returnsOpenNextMktres10 < 0].returnsOpenNextMktres10.max() * -1)

for (
    market_obs_df,
    news_obs_df,
    predictions_template_df,
) in env.get_prediction_days():
    predictions_df = (
        predictions_template_df.set_index("assetCode")
        .join(
            market_obs_df.set_index(["assetCode", "time"])
            .join(df.returnsOpenNextMktres10, how="left")
            .returnsOpenNextMktres10.reset_index()
            .drop("time", axis=1)
            .set_index("assetCode"),
            how="inner",
            lsuffix="_",
        )
        .reset_index()
    )
    predictions_df.confidenceValue = 1 / predictions_df.returnsOpenNextMktres10 * m
    predictions_df.drop("returnsOpenNextMktres10", axis=1)
    
    env.predict(predictions_df)

env.write_submission_file()