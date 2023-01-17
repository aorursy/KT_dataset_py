!pip install https://github.com/ceshine/shap/archive/master.zip
import sys
import gc
import torch.nn as nn
import torch
from torch.nn.utils.weight_norm import weight_norm
from sklearn import preprocessing
import shap
import numpy as np
import joblib
%ls ../input/data/cache/model_cache/
MODEL = "../input/data/cache/model_cache/snapshot_PUBG_0.02873547.pth"
class MLPModel(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.model = nn.Sequential(
            weight_norm(nn.Linear(num_features, 64)),
            nn.ELU(),
            weight_norm(nn.Linear(64, 64)),
            nn.ELU(),
            weight_norm(nn.Linear(64, 64)),
            nn.ELU(),
            weight_norm(nn.Linear(64, 64)),
            nn.ELU(),          
            weight_norm(nn.Linear(64, 1)),
        )

    def forward(self, input_tensor):
        return torch.clamp(self.model(input_tensor), 0, 1)
x_train, features = joblib.load("../input/x_train_dump.jl")
DEVICE = "cpu"
model = MLPModel(len(features)).to(DEVICE)
model.load_state_dict(torch.load(MODEL, map_location='cpu'))
%%time
e = shap.DeepExplainer(
        model, 
        torch.from_numpy(
            x_train[np.random.choice(np.arange(len(x_train)), 10000, replace=False)]
        ).to(DEVICE))
%%time
x_samples = x_train[np.random.choice(np.arange(len(x_train)), 300, replace=False)]
print(len(x_samples))
shap_values = e.shap_values(
    torch.from_numpy(x_samples).to(DEVICE)
)
shap_values.shape
import pandas as pd
df = pd.DataFrame({
    "mean_abs_shap": np.mean(np.abs(shap_values), axis=0), 
    "stdev_abs_shap": np.std(np.abs(shap_values), axis=0), 
    "name": features
})
df.sort_values("mean_abs_shap", ascending=False)[:10]
shap.summary_plot(shap_values, features=x_samples, feature_names=features)