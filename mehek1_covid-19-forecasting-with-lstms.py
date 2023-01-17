import torch
import numpy as np
np.random.seed(420)
import pandas as pd
!ls ../input
!mkdir models
train = pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv").replace(np.nan, 0)
train
COLUMNS_TO_DROP = ['Id', 'Province_State', 'Country_Region', 'Date']
from plotly import graph_objects as go
country_name = "India"
country = train[train['Country_Region'] == country_name]

fig = go.Figure()
fig.add_trace(go.Scatter(x=country['Date'], y=country['ConfirmedCases'], name='Confirmed Cases'))
fig.add_trace(go.Scatter(x=country['Date'], y=country['Fatalities'], name='Fatalities'))
fig.update_layout(title='Total COVID-19 in {}'.format(country_name))
fig.show()
def Cummulative_to_Absolute(df):
    df = df.copy()
    new_cases = [0]
    new_fatalities = [0]
    for i in range(1, len(df)):
        new_cases.append(df['ConfirmedCases'].iloc[i] - df['ConfirmedCases'].iloc[i-1])
        new_fatalities.append(df['Fatalities'].iloc[i] - df['Fatalities'].iloc[i-1])
    df['ConfirmedCases'] = new_cases
    df['Fatalities'] = new_fatalities
    return df

country_absolute = Cummulative_to_Absolute(country)
#print(sum(country_absolute['ConfirmedCases'].values))
fig = go.Figure()
fig.add_trace(go.Scatter(x=country_absolute['Date'], y=country_absolute['ConfirmedCases'], name='Confirmed Cases'))
fig.add_trace(go.Scatter(x=country_absolute['Date'], y=country_absolute['Fatalities'], name='Fatalities'))
fig.update_layout(title='Daily COVID-19 in {}'.format(country_name))
fig.show()
grouped = train.groupby('Country_Region')
countries = list(grouped.sum().index)
num_provinces = [(c, len(train[train['Country_Region'] == c])// 61) for c in countries]
num_provinces
long = train['Province_State'].to_list()
lat = train['Country_Region'].to_list()
unique_long_lat = list(set(zip(long, lat)))
len(unique_long_lat) # Unique Places where the stats were recorded
unique_long_lat
np.random.shuffle(unique_long_lat)
train_long_lat = unique_long_lat[:int(len(unique_long_lat) * 0.7)]
val_long_lat = unique_long_lat[int(len(unique_long_lat) * 0.7):]
# Function to split the Dataframe by longtitude and latitude
def make_df_by_long_lat(df, long_lat):
    #print(long_lat)
    one = df[df['Province_State'] == long_lat[0]]
    #print(one)
    two = one.loc[one['Country_Region'] == long_lat[1]]
    two = two.drop(columns=COLUMNS_TO_DROP)
    return two
make_df_by_long_lat(train, (0, "Afghanistan"))#train_long_lat[0])
#train
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

train_copy = train.drop(columns=COLUMNS_TO_DROP)
scaler.fit(train_copy.values)
def sliding_window(series, seq_len, scaler):
    series = scaler.transform(series)
    x = []
    y = []
    for i in range(len(series) - seq_len - 1):
        x.append(np.expand_dims(series[i:i+seq_len],  axis=0))
        y.append(np.expand_dims(series[i+seq_len], axis=0))
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    return x, y

country = country.drop(columns=COLUMNS_TO_DROP)
#print(india)
x, y = sliding_window(country, 14, scaler)
x.shape
y.shape
test = make_df_by_long_lat(train, (0, 'Uruguay'))
x, y = sliding_window(test, 14, scaler)
x.shape
def states_countries(df, seq_len, states_countries, scaler):
    x_final, y_final = [], []
    for state_country in states_countries:
        #print(state_country)
        x, y = sliding_window(Cummulative_to_Absolute(make_df_by_long_lat(df, state_country)), seq_len, scaler)
        #print(x, y)
        x_final.append(x)
        y_final.append(y)
    x_final = np.concatenate(x_final, axis=0)
    y_final = np.concatenate(y_final, axis=0)
    return x_final, y_final
x_train, y_train = states_countries(train, 14, train_long_lat, scaler)
x_val, y_val = states_countries(train, 14, val_long_lat, scaler)
print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
from torch.utils.data import TensorDataset
train_ds = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
val_ds = TensorDataset(torch.FloatTensor(x_val), torch.FloatTensor(y_val))
from torch import nn

class DoomsDayPredictor(nn.Module):
    def __init__(self, in_features, out_features, hidden_layers, hidden_dim):
        super(DoomsDayPredictor, self).__init__()
        
        self.lstm = nn.LSTM(in_features, num_layers=hidden_layers, hidden_size=hidden_dim, dropout=0.3, batch_first=True)
        
        self.fc1 = nn.Linear(hidden_dim, out_features)
        self.prelu = nn.PReLU()
        
    def reset_hidden(self):
        self.hidden = (torch.zeros(self.hidden[0].shape), torch.zeros(self.hidden[1].shape))
    
    def forward(self, x):
        out, self.hidden = self.lstm(x)
        #print(out.shape)
        out = self.prelu(out[:, -1, :])
        out = self.fc1(out)
        return out
model = DoomsDayPredictor(2, 2, 1, 5)
model
x_dummy, y_dummy = train_ds[0]
x_dummy.shape
y_dummy
y_pred = model(torch.FloatTensor(x_dummy.unsqueeze(0)))
y_pred.shape
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

metrics = {
    'r2': lambda y_pred, y_true: r2_score(y_true.cpu().numpy(), y_pred.cpu().numpy()),
    'mse': lambda y_pred, y_true: mean_squared_error(y_true.cpu().numpy(), y_pred.cpu().numpy()),
    'mae': lambda y_pred, y_true: mean_absolute_error(y_true.cpu().numpy(), y_pred.cpu().numpy()),    
}
def evaluate(model, dataloader, metrics):
    with torch.no_grad():
        y_pred = []
        y_true = []
        for x, y in dataloader:
            model.reset_hidden()
            
            out = model(x)

            y_pred.append(out)
            y_true.append(y)
            #print(out.shape, y.shape)
        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)

        computed = {} 
        for metric_name, metric in metrics.items():
            computed[metric_name] = metric(y_true, y_pred)
    return computed
def train_model(model, optimizer, train_dl, val_dl, metrics):
    
    patience = 5
    p = 0
    
    # Hyperparameters
    max_epochs = 50
    
    # Loss function
    loss_fn = nn.SmoothL1Loss(reduction='sum')
    loss_history = []
    
    evaluate_interval = 500
    steps = 0
    
    best_r2 = float("-inf")
    
    # flag for stopping train loop 
    flag = 0
    
    # training loop
    for ep in range(max_epochs):
        if flag == 1:
            break
        for x, y in train_dl:
            
            model.reset_hidden()
            
            optimizer.zero_grad()
            
            out = model(x)
            
            loss = loss_fn(y, out)
            loss_history.append(loss.item())
            
            loss_history = loss_history[-10:]
            avg_loss = sum(loss_history)/len(loss_history)
            
            loss.backward()
            optimizer.step()
            
            #if steps % 10:
            #    print(steps, loss.item(), avg_loss)#, optimizer.learning_rate)
            
            #if steps % 10:
            #    lr_scheduler.step()
                
            if steps % evaluate_interval == 0:
                #print(steps)
                val_metrics = evaluate(model, val_dl, metrics)
                if best_r2 < val_metrics['r2']:
                    best_r2 = val_metrics['r2']
                    torch.save(model, 'models/best.pth')
                else:
                    p += 1
                    if p >= patience:
                        flag = 1
                        print("Training stopped at epoch", ep)
                        break
            steps += 1
            
    return torch.load('models/best.pth'), best_r2
from torch.utils.data import DataLoader

batch_size = 32

tr_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=2)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
best, best_r2 = train_model(model, optimizer, tr_dl, val_dl, metrics)
print("Best Checkpoint Validation R^2 Value: ", best_r2)
india = train[train['Country_Region'] == 'India']
india = india.drop(columns=COLUMNS_TO_DROP)

two_weeks = torch.FloatTensor(scaler.transform(india.iloc[-14:, :])).unsqueeze(0)
two_weeks.shape
with torch.no_grad():
    y_pred = best(two_weeks).numpy()
    y_pred = scaler.inverse_transform(y_pred)
tomorrow = y_pred
num_days_to_predict = 7
country_name = "India"
days_to_show = 7 * 3

country = train[train['Country_Region'] == country_name]
today_cases = country['ConfirmedCases'].iloc[-1]
today_fatalities = country['Fatalities'].iloc[-1]

country = Cummulative_to_Absolute(country)
x_dates = country['Date'][-days_to_show:]
country = country.drop(columns=COLUMNS_TO_DROP)

two_weeks = torch.FloatTensor(scaler.transform(india.iloc[-14:, :])).unsqueeze(0)

# Predict for One Week
with torch.no_grad():
    for _ in range(num_days_to_predict):
        y_pred = best(two_weeks).unsqueeze(0)
        two_weeks = torch.cat([two_weeks, y_pred], dim=1)[:, -14:, :]
        #print(two_weeks.shape)

y_pred = scaler.inverse_transform(two_weeks.squeeze(0).numpy())
#print(y_pred.shape)
country = country.iloc[-days_to_show:, :]
fig = go.Figure()

x_new_dates = pd.date_range(start=x_dates.iloc[-1], periods=num_days_to_predict)

fig.add_trace(go.Scatter(x=x_dates, y=country['ConfirmedCases'], name='Confirmed Cases'))
fig.add_trace(go.Scatter(x=x_dates, y=country['Fatalities'], name='Fatalities'))
fig.add_trace(go.Scatter(x=x_new_dates, y=y_pred[:, 0], name='Predicted Confirmed Cases'))
fig.add_trace(go.Scatter(x=x_new_dates, y=y_pred[:, 1], name='Predicted Fatalities'))
fig.update_layout(title='Forecast for COVID-19 in {}'.format(country_name), xaxis_title="Date", yaxis_title="New Cases/Fatalities")
fig.show()
next_cases = int(today_cases + y_pred[:, 0].sum())
next_fatal = int(today_fatalities + y_pred[:, 1].sum())

print("Predictions")
print("                   Tomorrow          After {} Days".format(num_days_to_predict))
print("New Cases:        ", int(tomorrow[0][0]),"             ", next_cases)
print("Total Fatalities: ", int(tomorrow[0][1]),"              ", next_fatal)
