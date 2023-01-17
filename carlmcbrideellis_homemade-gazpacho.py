import pandas as pd

tomate = pd.read_csv("../input/-32-house-prices-version-7-score-007483/submission.csv")

pepino = pd.read_csv("../input/7-lines-of-code-to-reach-6th-place/submission.csv")
tomate["SalePrice"] = tomate["SalePrice"]*0.91 + pepino["SalePrice"]*0.09
tomate.to_csv('submission.csv', index=False)