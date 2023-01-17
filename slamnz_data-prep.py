from pandas import read_json

data = read_json("../input/roam_prescription_based_prediction.jsonl", lines=True)
data = data.sample(frac=0.01)
data.shape
from json import loads



cms_prescription_counts = data["cms_prescription_counts"]

provider_variables = data["provider_variables"]

npi = data["npi"]



from pandas import DataFrame



x1 = DataFrame(data=[row for row in cms_prescription_counts])

x2 = DataFrame(data=[row for row in provider_variables])

x3 = DataFrame(data=[row for row in npi])



from pandas import concat



data = concat([x1,x2,x3], axis = 1)