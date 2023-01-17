import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import datetime

import os

import glob

import pickle

import torch

import torch.nn as nn

from torch import optim

import torch.nn.functional as F

from IPython.display import Image

import warnings

warnings.filterwarnings("ignore")
# read both training and test data as Pandas Dataframe to inspect the data

forecasting_location = "/kaggle/input/covid19-global-forecasting-week-4/"

training_file = forecasting_location + "train.csv"

test_file = forecasting_location + "test.csv"

training_df = pd.read_csv(training_file)

test_df = pd.read_csv(test_file)
training_df.info
test_df.info
training_countries = training_df["Country_Region"].unique()

test_countries = test_df["Country_Region"].unique()

print("Number of countries in Training File: {0}".format(len(training_countries)))

print("Number of countries in Test File: {0}".format(len(test_countries)))
# Distritbution of Entries (# of Occurrences) by country in training file (Top Countries)

def top_countries_by_entries(df, title, top_k = 20):

    df_country_groupby_size = df.groupby("Country_Region").count().rename(columns = {"Date": "Count"})[["Count"]].sort_values(by = ["Count"], ascending = False)

    df_country_groupby_size.head(top_k).plot.bar(title = title)

    

top_countries_by_entries(training_df, "Top 20 Countries in Training File")
def filter_by_country(df, country):

    return df[df["Country_Region"] == country]



training_df_usa = filter_by_country(training_df, "US")

training_df_china = filter_by_country(training_df, "China")

training_df_italy = filter_by_country(training_df, "Italy")

training_df_spain = filter_by_country(training_df, "Spain")

training_df_france = filter_by_country(training_df, "France")

training_df_germany = filter_by_country(training_df, "Germany")

training_df_iran = filter_by_country(training_df, "Iran")

training_df_turkey = filter_by_country(training_df, "Turkey")

training_df_japan = filter_by_country(training_df, "Japan")

training_df_singapore = filter_by_country(training_df, "Singapore")

training_df_australia = filter_by_country(training_df, "Australia")
def trend_visualization_per_country(df_country, country, ms = 4):

    df_country = df_country.sort_values(by = ["Date"])

    

    # need to sum numbers of all provinces/states together for each date

    df_country = df_country.groupby("Date")["ConfirmedCases", "Fatalities"].agg("sum")

    

    dates = df_country.index

    dates = [datetime.datetime(int(date[:4]), int(date[5:7]), int(date[8:10]), 0, 0) for date in dates]

    confirmed_cases = df_country["ConfirmedCases"]

    fatalities = df_country["Fatalities"]

    

    

    fig, ax = plt.subplots(figsize=(20,8))

    ax.plot_date(dates, confirmed_cases, ms = ms, label = "Confirmed Cases")

    ax.plot_date(dates, fatalities, ms = ms, label = "Fatalities")

    ax.legend()

    ax.set_title("Confirmed Cases & Fatalities in {0}".format(country))
# Trend in USA

trend_visualization_per_country(training_df_usa, "USA")
# Trend in China

trend_visualization_per_country(training_df_china, "China")
# Trend in Italy

trend_visualization_per_country(training_df_italy, "Italy")
# Trend in Spain

trend_visualization_per_country(training_df_spain, "Spain")
# Trend in France

trend_visualization_per_country(training_df_france, "France")
# Trend in Germany

trend_visualization_per_country(training_df_germany, "Germany")
# Trend in Iran

trend_visualization_per_country(training_df_iran, "Iran")
# Trend in Turkey

trend_visualization_per_country(training_df_turkey, "Turkey")
# Trend in Japan

trend_visualization_per_country(training_df_germany, "Japan")
# Trend in Singapore

trend_visualization_per_country(training_df_germany, "Singapore")
# Trend in Australia

trend_visualization_per_country(training_df_australia, "Australia")
# since there are 185 countries, I choose the following 11 countries

countries = ["China", "US", "Italy", "Spain", "France", "Germany", "Iran", "Turkey", "Japan", "Singapore", "Australia"]

metrics = ["ConfirmedCases", "Fatalities"]
def trend_visualization_per_metric(df, metrics, countries, ms = 1):

    df = df.sort_values(by = ["Date"])

    

    for metric in metrics:

        fig, ax = plt.subplots(figsize=(20,8))

        

        for country in countries:

            df_country = df[df["Country_Region"] == country]

        

            # need sum all provinces together for each date

            df_country = df_country.groupby("Date")["ConfirmedCases", "Fatalities"].agg("sum")

    

            #print(df_country.tail())

    

            dates = df_country.index

            dates = [datetime.datetime(int(date[:4]), int(date[5:7]), int(date[8:10]), 0, 0) for date in dates]

            values = df_country[metric]

    

            ax.plot(dates, values, ms = ms, label = country)

            

        ax.legend()

        ax.set_title("Trend of {0}".format(metric))
trend_visualization_per_metric(training_df, metrics, countries)
# construct a country-to-state dictionary

all_countries = training_df["Country_Region"].unique().tolist()

all_locations = {}

for country in all_countries:

    df_country = training_df[training_df["Country_Region"] == country]

    all_states = df_country["Province_State"].unique().tolist()

    for state in all_states:

        if country in all_locations:

            tmp = all_locations[country]

            tmp.append(str(state))

            all_locations[country] = tmp

        else:

            all_locations[country] = [str(state)]

            

# partition training set by each unique (Country_Region, Province_State)

for country in all_locations:

    os.system("mkdir -p \"partition/{0}\"".format(country))

    states = all_locations[country]

    country_df = training_df[training_df["Country_Region"] == country]

    for state in states:

        location = "partition/{0}/{1}.csv".format(country, state)

        if state != "nan":

            location_df = country_df[country_df["Province_State"] == state]

            location_df.to_csv(location, index = False)

        else:

            country_df.to_csv(location, index = False)
Image("../input/images/architecture.png")
# define encoder & decoder



class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size):

        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size

        self.linear = nn.Linear(input_size, hidden_size)

        self.gru = nn.GRU(hidden_size, hidden_size)

        

        # use glorot initialization to avoid loss plateauing

        nn.init.xavier_normal_(self.linear.weight)



    def forward(self, input_data, hidden):

        # GRU requires 3-dimensional inputs

        input_data_transformed = self.linear(input_data).view(-1, 1, self.hidden_size)

        hidden = hidden.view(-1, 1, self.hidden_size)

        

        output, hidden = self.gru(input_data_transformed, hidden)

        return output, hidden



    def initHidden(self):

        return torch.zeros(1, self.hidden_size)



    

class DecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size):

        super(DecoderRNN, self).__init__()

        self.hidden_size = hidden_size

        self.gru = nn.GRU(hidden_size, hidden_size)

        self.out = nn.Linear(hidden_size, output_size)

        

        # use glorot initialization to avoid loss plateauing

        nn.init.xavier_normal_(self.out.weight)



    def forward(self, input_data, hidden):

        # GRU requires 3-dimensional inputs

        input_data = input_data.view(-1, 1, self.hidden_size)

        hidden = hidden.view(-1, 1, self.hidden_size)

        

        output, hidden = self.gru(input_data, hidden)

        output = F.relu(self.out(output))

        return output, hidden
# a function to train an encoder-decoder model for a given (Country_Region, Province_State)



# Arguments:

# - train_csv: the csv file for a (Country_Region, Province_State)

# - epochs: number of epochs to train

# - frequency: how often (in terms of epochs) to plot the predicted values vs true numbers

# - which_feature: 0 means ConfirmedCases, 1 means Fatalities

# - val_frac: fraction of the training data as validation data

# - hidden_size: the hidden size for the encoder and the decoder

# - lr: learning rate for the SGD optimizer

# - remove_zero: whether to remove 0 values in the training data

# - normalize: whether to z-score normalize the training data

# - log: a log file location



# Return:

# - True: if the model performance plateaus, so the client code can retry

# - False: the model performance doesn't plateau, so the client code can proceed to the next model



def train_encoder_decoder(train_csv, epochs = 10, frequency = 5, which_feature = 0, val_frac = 0.1, hidden_size = 20, lr = 1e-3, remove_zero = False, normalize = False, log = "encoder_decoder/training.log"):

    country = train_csv.split("/")[-2]

    state = train_csv.split("/")[-1][:-4]

    

    if which_feature == 0:

        feature = "ConfirmedCases"

    else:

        feature = "Fatalities"

        

    # serializing the models

    encoder_location = "encoder_decoder/{0}/{1}/{2}_encoder.mdl".format(country, state, feature)

    decoder_location = "encoder_decoder/{0}/{1}/{2}_decoder.mdl".format(country, state, feature)

    

    # serializing the encoder and decoder hidden states after training

    decoder_hidden_location = "encoder_decoder/{0}/{1}/{2}_decoder_hidden.p".format(country, state, feature)

    encoder_hidden_location = "encoder_decoder/{0}/{1}/{2}_encoder_hidden.p".format(country, state, feature)

    

    # serializing the mean and std-dev of the data for prediction, if normalize = True

    stats_location = "encoder_decoder/{0}/{1}/{2}_stats.p".format(country, state, feature)

    

    # make directory for saving

    os.system("mkdir -p \"encoder_decoder/{0}/{1}\"".format(country, state))

    

    df = pd.read_csv(train_csv)

    

    x_mean = None

    x_std = None

    

    x = df[feature].values

    

    if remove_zero:

        x = x[x != 0]

        

    if normalize:

        # z-score normalize

        # normalize because we are interested in the relative trend, not the absolute numbers

        # normalization turns out to be more effective than no normalization as the optimization is more sensitive and thus less stable to larger numbers

        x_mean = x.mean()

        x_std = x.std()

        x = (x - x_mean) / x_std

        

    if (x_mean == 0) and (x_std == 0):

        print("Skipping this feature because all data are 0")

        return

        

    # save normalization stats for prediction    

    stats = {"mean": x_mean, "std": x_std}

    with open(stats_location, 'wb') as fp:

        pickle.dump(stats, fp, protocol=pickle.HIGHEST_PROTOCOL)



    size = len(x)

    

    train_size = int(size * (1 - val_frac))

    val_size = size - train_size

    

    x_train = x[: train_size]

    x_val = x[train_size :]

    

    # initialize encoder and decoder

    encoder = EncoderRNN(input_size = 1, hidden_size = hidden_size)

    decoder = DecoderRNN(hidden_size = hidden_size, output_size = 1)

    

    # use mean squared error as loss

    criterion = nn.MSELoss()

    

    # optimizers

    encoder_optimizer = optim.SGD(encoder.parameters(), lr = lr)

    decoder_optimizer = optim.SGD(decoder.parameters(), lr = lr)

    

    losses = []

    plateau = None

    

    for epoch in range(epochs):

        # zero the gradients of the optimizers

        encoder_optimizer.zero_grad()

        decoder_optimizer.zero_grad()

    

        # loss

        loss = 0

        val_loss = 0

        

        # get an initial hidden state (0 vector) for the encoder at the beginning of each epoch

        encoder_hidden = encoder.initHidden()

        

        # for storing the predictions

        decoder_outputs_train = torch.zeros(train_size - 1)

        decoder_outputs_val = torch.zeros(val_size)

    

        # training mode

        encoder.train()

        decoder.train()

        for index, target in enumerate(x_train[1:]):

            

            # encode the input (previous day's number) into the encoded vector

            encoder_output, encoder_hidden = encoder(torch.tensor(x[index - 1]).view(1, 1), encoder_hidden)

            

            # create an artifact (0 vector) for the decoder - since we don't have other input features

            decoder_input = torch.zeros(1, encoder.hidden_size)

    

            # set the decoder hidden state to the final encoder hidden state (encoded vector) (accumulated over all inputs up till this point)

            decoder_hidden = encoder_hidden

    

            # pass through decoder

            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

            

            decoder_output = decoder_output.squeeze()

    

            loss += criterion(decoder_output, torch.tensor(target)) * (index + 1)

            # more weights/emphasis for recent dates

            # this is the key for emphasizing on the most recent developments

        

            decoder_outputs_train[index] = decoder_output

        

        loss /= (train_size - 1) # since we didn't encoder the number from the last day

        

        losses.append(loss.detach().numpy().tolist())

        

        if (epoch + 1) % frequency == 0:

            

            plateau = np.array(losses).std()

            

            # the loss of the model depends heavily on weight initialization, sometimes the loss never improves and the prediction curve is flat,

            # in which case we need to start over

            if (plateau <= 1e-4):

                print("Loss Standard Deviation for the last {} Training Epochs is {}. The model performance plateaus, will retry!".format(frequency, plateau))



                # lets the client code know that this model performance plateaus, so it can retry

                return True

            

            # print training loss at this epoch

            print("Epoch {0} - Training Loss: {1}".format(epoch + 1, loss))

            

            # plot prediction vs true value

            plot_pred_vs_true(decoder_outputs_train.squeeze(), torch.tensor(x_train[1:]).squeeze(), epoch, title = "encoder_decoder/{0}/{1}/{2}_prediction_vs_target_epoch_{3}_training.jpg".format(country, state, feature, epoch + 1), country = country, state = state, feature = feature)

            

            losses = []

        

        # optimize

        loss.backward()

        encoder_optimizer.step()

        decoder_optimizer.step()

        

        # eval mode

        encoder.eval()

        decoder.eval()

        

        # using the last day's number in training as the starting input for validation

        previous_number = x_train[-1]

        

        for index, target in enumerate(x_val):

            

            # encode the input into the encoded vector

            encoder_output, encoder_hidden = encoder(torch.tensor(previous_number).view(1, 1), encoder_hidden)

            

            # create an artifact (0 vector) for the decoder - since we don't have other input features

            decoder_input = torch.zeros(1, encoder.hidden_size)

    

            # set the decoder hidden state to the final encoder hidden state (encoded vector) (accumulated over all inputs up till this point)

            decoder_hidden = encoder_hidden

    

            # pass through decoder

            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

            

            decoder_output = decoder_output.squeeze()

    

            val_loss += criterion(decoder_output, torch.tensor(target))

        

            decoder_outputs_val[index] = decoder_output

            

            # update the input to always be the previous day's number

            previous_number = target

        

        val_loss /= val_size

        

        if (epoch + 1) % frequency == 0:

            print("Epoch {0} - Validation Loss: {1}".format(epoch + 1, val_loss))

            # plot prediction vs true value

            plot_pred_vs_true(decoder_outputs_val.squeeze(), torch.tensor(x_val).squeeze(), epoch, title = "encoder_decoder/{0}/{1}/{2}_prediction_vs_target_epoch_{3}_val.jpg".format(country, state, feature, epoch + 1), country = country, state = state, mode = "Validation", feature = feature)

    

    with open(decoder_hidden_location, 'wb') as fp:

        pickle.dump(decoder_hidden, fp, protocol=pickle.HIGHEST_PROTOCOL)

            

    # save encoded vector for testing/prediction

    with open(encoder_hidden_location, 'wb') as fp:

        pickle.dump(encoder_hidden, fp, protocol=pickle.HIGHEST_PROTOCOL)

        

    if (epoch + 1 == epochs):

        if (state != "nan"):

            os.system("echo \"Training on {} for {} {}\" >> {}".format(feature, country, state, log))

        else:

            os.system("echo \"Training on {} for {}\" >> {}".format(feature, country, log))

        os.system("echo \"Training Loss for Epoch {}: {}\" >> {}".format(epoch + 1, loss, log))

        os.system("echo \"Validation Loss for Epoch {}: {}\" >> {}".format(epoch + 1, val_loss, log))

        

    torch.save(encoder, encoder_location)

    torch.save(decoder, decoder_location)

    

    os.system("echo >> {}".format(log))

    

    # reaches the end, so the model performance doesn't plateau and the client code can proceed to the next model

    return False

    

# a function to plot the predicted numbers vs the real numbers in an given epoch during training

def plot_pred_vs_true(pred, true, epoch, title, country, state, feature = "ConfirmedCases", mode = "Training", ms = 5):

    pred = pred.squeeze().detach().numpy()

    true = true.squeeze()

    size = len(pred)

    x = list(range(size))

    fig, ax = plt.subplots(figsize=(10,7))

    

    ax.plot(x, pred, ms = ms, label = "prediction")

    ax.plot(x, true, ms = ms, label = "true")

    

    if (state == "nan"):

        state = ""

    else:

        state = ", {}".format(state)

            

    ax.legend()

    ax.set_title("{} Epoch {} for {} in {}{}".format(mode, epoch + 1, feature, country, state))

    plt.savefig(title)

    plt.show()
# train encoder and decoder for these 4 (Country_Region, Province_State) locations



locations = [("US", "New York"), ("China", "Hubei"), ("US", "New Jersey"), ("Italy", "nan")]



path = "/kaggle/working/partition"



features = {0: "ConfirmedCases", 1: "Fatalities"}



os.system("rm -fr encoder_decoder/")



for (country, state) in locations:

    if (state != "nan"):

        print("Modeling for {}, {}".format(country, state))

    else:

        print("Modeling for {}".format(country))

    csv_file = path + "/{}/{}.csv".format(country, state)

    for which_feature in [0, 1]:

        print(">> Metric/Feature: {}".format(features[which_feature]))

        trial = 0

        plateau = True

        while plateau:

            trial += 1

            print(">> >> Trial {}".format(trial))

            plateau = train_encoder_decoder(csv_file, lr = 1e-3, epochs = 100, frequency = 10, normalize = True, which_feature = which_feature)
!cat encoder_decoder/training.log
Image("encoder_decoder/US/New York/ConfirmedCases_prediction_vs_target_epoch_100_training.jpg")
Image("encoder_decoder/US/New York/ConfirmedCases_prediction_vs_target_epoch_100_val.jpg")
Image("encoder_decoder/US/New York/Fatalities_prediction_vs_target_epoch_100_training.jpg")
Image("encoder_decoder/US/New York/Fatalities_prediction_vs_target_epoch_100_val.jpg")
Image("encoder_decoder/China/Hubei/ConfirmedCases_prediction_vs_target_epoch_100_training.jpg")
Image("encoder_decoder/China/Hubei/ConfirmedCases_prediction_vs_target_epoch_100_val.jpg")
Image("encoder_decoder/China/Hubei/Fatalities_prediction_vs_target_epoch_100_training.jpg")
Image("encoder_decoder/China/Hubei/Fatalities_prediction_vs_target_epoch_100_val.jpg")
Image("encoder_decoder/US/New Jersey/ConfirmedCases_prediction_vs_target_epoch_100_training.jpg")
Image("encoder_decoder/US/New Jersey/ConfirmedCases_prediction_vs_target_epoch_100_val.jpg")
Image("encoder_decoder/US/New Jersey/Fatalities_prediction_vs_target_epoch_100_training.jpg")
Image("encoder_decoder/US/New Jersey/Fatalities_prediction_vs_target_epoch_100_val.jpg")
Image("encoder_decoder/Italy/nan/ConfirmedCases_prediction_vs_target_epoch_100_training.jpg")
Image("encoder_decoder/Italy/nan/ConfirmedCases_prediction_vs_target_epoch_100_val.jpg")
Image("encoder_decoder/Italy/nan/Fatalities_prediction_vs_target_epoch_100_training.jpg")
Image("encoder_decoder/Italy/nan/Fatalities_prediction_vs_target_epoch_100_val.jpg")
locations = [("US", "New York"), ("China", "Hubei"), ("US", "New Jersey"), ("Italy", "nan")]



csv_path = "/kaggle/working/partition/{}/{}.csv"

encoder_decoder_path = "/kaggle/working/encoder_decoder/{}/{}"



features = {0: "ConfirmedCases", 1: "Fatalities"}
# training data stops on 04-29-2020, so testing starts on 04-30-2020

# but the last encoded date in training is actually 04-28-2020 (04-29-2020 is never encoded in training so we treat 04-29-2020 also as unseen data) so testing actually starts on 04-29-2020 and ends on 2020-05-14 (16 days)



index_to_date = {}

for index in range(2):

    index_to_date[index] = "2020-04-{}".format(index + 29)

for index in range(2, 11):

    index_to_date[index] = "2020-05-0{}".format(index - 1)

for index in range(11, 16):

    index_to_date[index] = "2020-05-{}".format(index - 1)

print(index_to_date)
# configurations from training

normalize = True

remove_zero = False

frequency = 10

epochs = 100

hidden_size = 20



os.system("rm -fr prediction")



# recover the original/authentic numeric range

def unnormalize(x, mean, std):

    return x * std + mean



for (country, state) in locations:

    for feature in features:

        feature = features[feature]

        

        training_csv = csv_path.format(country, state)

    

        df = pd.read_csv(training_csv)

    

        if (state != "nan"):

            print("Predicting {} for {}, {}".format(feature, country, state))

        else:

            print("Predicting {} for {}".format(feature, country))

    

        encoder = encoder_decoder_path.format(country, state) + "/{}_encoder.mdl".format(feature)

        decoder = encoder_decoder_path.format(country, state) + "/{}_decoder.mdl".format(feature)

    

        encoder_hidden = encoder.replace(".mdl", "_hidden.p")

        stats = decoder.replace("_decoder.mdl", "_stats.p")

    

        prediction_csv = "prediction/{0}/{1}/{2}.csv".format(country, state, feature)

        graph_location = "prediction/{0}/{1}/{2}.jpg".format(country, state, feature)

        os.system("mkdir -p \"prediction/{0}/{1}\"".format(country, state))

    

        # load trained models from training

        decoder = torch.load(decoder)

        encoder = torch.load(encoder)

    

        # load encoded vector from training

        with open(encoder_hidden, 'rb') as fp:

            encoder_hidden = pickle.load(fp)

        

        # load mean and std-dev for un-normalization

        with open(stats, 'rb') as fp:

            stats = pickle.load(fp)

        

        # encode 04-29 data from training csv as it's never been encoded in training

        if (state != "nan"):

            encoder_input = df[(df["Country_Region"] == country) & (df["Province_State"] == state) & (df["Date"] == "2020-04-29")][feature].values

        else:

            encoder_input = df[(df["Country_Region"] == country) & (df["Province_State"].isnull()) & (df["Date"] == "2020-04-29")][feature].values

        encoder_input = torch.tensor(encoder_input).view(1, -1).float()

    

        encoder_output, encoder_hidden = encoder(encoder_input, encoder_hidden)

        

        decoder_hidden = encoder_hidden

        

        values = []

    

        for index in index_to_date:

            if index == 0:

                # 2020-04-29 already encoded above

                continue

        

            # create a start-of-sequence tensor for the decoder

            decoder_input = torch.zeros(1, hidden_size)

    

            # pass through decoder

            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

        

            # decoded/predicted outputs

            values.append(decoder_output.squeeze().detach().numpy().tolist())

        

            # accumulate the history by encoding the predicted number

            encoder_hidden = decoder_hidden

            encoder_output, encoder_hidden = encoder(decoder_output, encoder_hidden)

            decoder_hidden = encoder_hidden

    

        values = np.array(values)

    

        # un-normalize

        if normalize:

            values = unnormalize(values, stats["mean"], stats["std"])

    

        # plot prediction and save graph

        fig, ax = plt.subplots(figsize=(10,7))

    

        dates = [datetime.datetime(int(date[:4]), int(date[5:7]), int(date[8:10]), 0, 0) for date in list(index_to_date.values())[1:]]   

        ax.plot_date(dates, values, ms = 5)

        if (state != "nan"):

            ax.set_title("Predicting {} for {}, {}".format(feature, country, state))

        else:

            ax.set_title("Predicting {} for {}".format(feature, country))

        plt.savefig(graph_location)

        #plt.show()

        plt.close()

    

        df = pd.DataFrame(list(index_to_date.values())[1:], columns = ["Date"]) # skip the 1st day (04-29-2020) since it's afterall in the training data

        df["Province_State"] = state

        df["Country_Region"] = country

        df[feature] = values

    

        df.to_csv(prediction_csv, index = False)
Image("prediction/US/New York/ConfirmedCases.jpg")
Image("prediction/US/New York/Fatalities.jpg")
Image("prediction/China/Hubei/ConfirmedCases.jpg")
Image("prediction/China/Hubei/Fatalities.jpg")
Image("prediction/US/New Jersey/ConfirmedCases.jpg")
Image("prediction/US/New Jersey/Fatalities.jpg")
Image("prediction/Italy/nan/ConfirmedCases.jpg")
Image("prediction/Italy/nan/Fatalities.jpg")