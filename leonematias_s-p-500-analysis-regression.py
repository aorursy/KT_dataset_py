# Imports
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import keras
from keras import backend as K
import datetime as dt
from IPython.display import display, HTML
from sklearn.model_selection import train_test_split
import os

RAND_SEED = 1
np.random.seed(RAND_SEED)
# Load SPY ETF data from Kaggle data set
def load_symbol_df(dir_path, file_name):
    full_path = dir_path + file_name
    if os.stat(full_path).st_size > 0: 
        df = pd.read_csv(full_path)
        columns = list(df.columns.values)
        symbol = file_name.split(".")[0]
        df["Symbol"] = "sp500"
        columns.insert(0, "Symbol")
        return df[columns]
    
df = load_symbol_df("../input/Data/ETFs/", "spy.us.txt")
df = df.sort_values(by="Date")
print("Dates: {}, from: {}, to: {}".format(len(df), df["Date"].min(), df["Date"].max()))
# All functions

def prices_to_samples(prices, windowLength, predDaysForward):
    X_data = []
    Y_data = []
    totalSamples = 0
    endIdx = len(prices) - windowLength + 1 - predDaysForward    
    for i in range(0, endIdx):
        X, Y = to_XY(prices, i, windowLength, predDaysForward)
        X_data.append(X)
        Y_data.append(Y)
        totalSamples += 1
    return (np.array(X_data).reshape((totalSamples, windowLength)), np.array(Y_data).reshape((totalSamples, predDaysForward)))

def prices_to_percentage(prices, base_price):
    base_price = max(base_price, 0.001)
    return ((prices - base_price) / base_price).astype("float32")

def percentage_to_prices(percentages, base_price):
    base_price = max(base_price, 0.001)
    return (base_price * (1.0 + percentages)).astype("float32")

def to_X(prices, startIdx, windowLength):
    winEndIdx = startIdx + windowLength
    return prices_to_percentage(prices[startIdx:winEndIdx], prices[winEndIdx - 1])
    #return prices[startIdx:winEndIdx].astype("float32")

def to_Y(prices, startIdx, windowLength, predDaysForward):
    predStartIdx = startIdx + windowLength
    predEndIdx = predStartIdx + predDaysForward
    return prices_to_percentage(prices[predStartIdx:predEndIdx], prices[predStartIdx - 1])
    #return prices[predStartIdx:predEndIdx].astype("float32")
    
def to_XY(prices, startIdx, windowLength, predDaysForward):
    return (to_X(prices, startIdx, windowLength), to_Y(prices, startIdx, windowLength, predDaysForward))

def prices_to_train_test_set(prices, trainSetProportion, windowLength, predDaysForward):
    X, Y = prices_to_samples(prices, windowLength, predDaysForward)
    return train_test_split(X, Y, test_size=1-trainSetProportion, random_state=RAND_SEED)

def plot_training_progress(history):
    y_loss = np.array(history.history["loss"]) * 100.0
    y_val_loss = np.array(history.history["val_loss"]) * 100.0
    x = np.arange(len(y_loss))
    fig, ax = plt.subplots(1,1, figsize=(20,8))
    plt.scatter(x, y_loss, c="red", label="Train loss")
    plt.plot(x, y_loss, color="red")
    plt.scatter(x, y_val_loss, c="blue", label="Val loss")
    plt.plot(x, y_val_loss, color="blue")
    plt.title("Training progress")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def test_model(prices, windowLength, predDaysForward, epochs):
    print("Testing model with windowLength: {} and predDaysForward: {}".format(windowLength, predDaysForward))
    
    # Create train/test set
    print("Creating samples...")
    X_train, X_test, Y_train, Y_test = prices_to_train_test_set(prices, 0.8, windowLength, predDaysForward)
    print("Samples - Total: {}, Train:{}, Test: {}".format(len(Y_train) + len(Y_test), len(Y_train), len(Y_test)))
    
    # Create Keras model
    model = keras.models.Sequential([
        keras.layers.Dense(32, input_shape=(windowLength,), activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(predDaysForward)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    
    # Callbacks
    def print_progress(epoch, logs):
        if epoch % 10 == 0:
            loss = logs["loss"]*100
            val_loss = logs["val_loss"]*100
            print("\tEpoch {}, Loss: {}%, Val Loss: {}%".format(epoch, round(loss,6), round(val_loss,6)))
        
    model_base_path = "models/"
    if not os.path.exists(model_base_path):
        os.makedirs(model_base_path)
    model_path = model_base_path + "sp500reg_history{}_forward{}".format(windowLength, predDaysForward)
    fit_callbacks = [
        keras.callbacks.ModelCheckpoint(model_path, monitor="loss", save_weights_only=True),
        keras.callbacks.LambdaCallback(on_epoch_end = lambda epoch, logs: print_progress(epoch, logs))
    ]
    
    # Train model
    print("Training...")
    BATCH_SIZE = 128
    history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=epochs, verbose=0, callbacks=fit_callbacks, validation_data=(X_test, Y_test))
    print("Training done")
    
    # Get predictions
    print("Predicting...")
    score_train = model.evaluate(X_train, Y_train, verbose=0)
    score_test = model.evaluate(X_test, Y_test, verbose=0)
    print("Train loss: {}%, Test loss: {}%".format(round(score_train*100, 6), round(score_test*100, 6)))
    
    # Save results
    res = {
        "model": model,
        "windowLength": windowLength,
        "predDaysForward": predDaysForward,
        "score_train": score_train,
        "score_test": score_test
    }
    
    # Plot training
    plot_training_progress(history)
    
    return res

def split_prices_by_date(df, date_threshold):
    return (df[df["Date"] < date_threshold]["Close"].values, df[df["Date"] >= date_threshold]["Close"].values)

# Function to show predicted return for specific symbol/date
def predict_return_by_date(symbol, prices_df, date, windowLength, predDaysForward, model, show_history=1.0):
    plot_num_hist_dates = int(windowLength * show_history)
    
    symbol_prices_df = prices_df[prices_df.Symbol == symbol][["Date", "Close"]].sort_values(by="Date")
    all_dates = symbol_prices_df.Date.values
    all_prices = symbol_prices_df.Close.values
    
    # Search current date
    current_date_search = np.nonzero(all_dates == date)[0]
    if len(current_date_search) == 0:
        print("No data for date: {}. Next business day: {}".format(date, next_business_day(date)))
        return
    current_date_idx = current_date_search[0]
    
    # Search first and prediction date
    first_date_idx = current_date_idx - windowLength + 1
    if first_date_idx < 0:
        print("Not enough historical data for date: {}. Oldest date: {}. Next business day: {}",format(date, all_dates[0], next_business_day(date)))
        return
    
    # Check if we have historical data to validate prediction
    last_pred_date_idx = current_date_idx + predDaysForward
    has_pred_date = last_pred_date_idx < len(all_dates)
    last_pred_date = all_dates[last_pred_date_idx] if has_pred_date else next_business_days(date, predDaysForward)[predDaysForward-1]
    
    # Predict future values
    X = to_X(all_prices, first_date_idx, windowLength).reshape((1, windowLength))
    Y_pred = model.predict(X)[0]
    Y_pred_prices = percentage_to_prices(Y_pred, all_prices[current_date_idx])
    Y_real_prices = all_prices[current_date_idx + 1 : last_pred_date_idx + 1] if has_pred_date else np.zeros(predDaysForward)
    mse = np.power(Y_real_prices - Y_pred_prices, 2).sum() / len(Y_real_prices)

    # Print summary
    print("{} - Current date: {}, Historical data since: {} ({} days), Price: ${}".format(symbol, date, all_dates[first_date_idx], windowLength, round(all_prices[current_date_idx],2)))
    print("Prediction for +{} days - {}: ${}".format(predDaysForward, last_pred_date, round(Y_pred_prices[-1],2)))
    print("Error: ${}".format(np.round(mse,4)))
    
    
    # Gather results
    report_data = []
    report_data.append([date, round(all_prices[current_date_idx],2), "0", 0])
    for i in range(0, predDaysForward):
        date_idx = current_date_idx + 1 + i
        next_date = all_dates[date_idx] if has_pred_date else next_business_days(date, i + 1)[i - 1]
        report_data.append([next_date, 
                            round(Y_real_prices[i],2), 
                            round(Y_pred_prices[i],2), 
                            round(Y_real_prices[i] - Y_pred_prices[i],2), 
                           ])
    report_df = pd.DataFrame(report_data, columns=["Date", "Price", "Prediction", "Diff"]) 
    
    # Plot historical prices
    fig, ax = plt.subplots(1,1, figsize=(20,8))
    plot_first_date_idx = current_date_idx - plot_num_hist_dates + 1
    plot_labels = np.append(all_dates[plot_first_date_idx : current_date_idx + 1], next_business_days(date, predDaysForward))
    #ax.set_xticklabels(plot_labels, rotation='horizontal')
    x_plot_hist = np.arange(plot_first_date_idx, current_date_idx + 1) - plot_first_date_idx
    y_plot_hist = all_prices[plot_first_date_idx : current_date_idx + 1]
    plt.plot(x_plot_hist, y_plot_hist, color="blue")
    plt.scatter(x_plot_hist, y_plot_hist, c="blue")
    
    # Plot vertical line to delimit historic vs prediction
    plt.axvline(x=current_date_idx - plot_first_date_idx, linestyle="--", color="gray")
    
    # Plot prediction
    x_plot_pred = np.arange(current_date_idx + 1, last_pred_date_idx + 1) - plot_first_date_idx
    plt.plot(x_plot_pred, Y_pred_prices, color="red")
    plt.scatter(x_plot_pred, Y_pred_prices, c="red")
    plt.plot([current_date_idx, current_date_idx+1]-plot_first_date_idx, [all_prices[current_date_idx], Y_pred_prices[0]], color="red", linestyle=":")
    
    # Plot actual future values (if available)
    if has_pred_date:
        plt.plot(x_plot_pred, Y_real_prices, color="green")
        plt.scatter(x_plot_pred, Y_real_prices, c="green")
        plt.plot([current_date_idx, current_date_idx+1]-plot_first_date_idx, all_prices[current_date_idx:current_date_idx+2], color="green", linestyle=":")
    
    # Show all
    plt.show()
    display(HTML(report_df.to_html(index=False) ))

    
def next_business_day(date_str):
    date_format = "%Y-%m-%d"
    date = dt.datetime.strptime(date_str, date_format)
    shift = dt.timedelta(1 + ((date.weekday()//4)*(6-date.weekday())))
    return dt.datetime.strftime(date + shift, date_format)

def next_business_days(date_str, n):
    days = []
    for i in range(n):
        date_str = next_business_day(date_str)
        days.append(date_str)
    return days


# Backtest a given model for a periods of dates
def backtest(df, since_date, windowLength, predDaysForward, model):
    # Find date to start evaluation
    first_eval_date = df[df["Date"] >= since_date].iloc[0]["Date"]
    first_eval_date_idx = df[df["Date"] == first_eval_date].index.item()
    all_dates = df.Date.values
    all_prices = df.Close.values
    total_dates_to_eval = len(all_dates) - first_eval_date_idx - predDaysForward
    
    # Create samples
    X = np.zeros((total_dates_to_eval, windowLength), dtype="float32")
    samples_ref = []
    for i in range(0, total_dates_to_eval):
        eval_idx = first_eval_date_idx + i
        last_win_idx = eval_idx - predDaysForward
        start_win_idx = last_win_idx - windowLength + 1
        X[i] = to_X(all_prices, start_win_idx, windowLength)
        samples_ref.append({
            "prev_date": all_dates[last_win_idx],
            "eval_date": all_dates[eval_idx],
            "prev_price": all_prices[last_win_idx],
            "eval_price": all_prices[eval_idx],
            "price_movement": all_prices[eval_idx] - all_prices[last_win_idx]
        })
        
    # Predict
    Y_pred = model.predict(X)
    
    # Check results for each day
    report_data = []
    y_plot_pred = np.zeros(total_dates_to_eval)
    y_plot_real = np.zeros(total_dates_to_eval)
    for i in range(total_dates_to_eval):
        ref = samples_ref[i]
        pred_return = Y_pred[i][0]
        prev_price = ref["prev_price"]
        real_price = ref["eval_price"]
        pred_price = prev_price * (1.0 + pred_return)
        price_diff = pred_price - real_price
        pred_return = (pred_price - prev_price) / prev_price * 100.0
        action = "Buy" if pred_return > 0.5 else "Nothing"
        profit = real_price - prev_price if action == "Buy" else 0
        report_data.append([ref["prev_date"], ref["eval_date"], round(prev_price,2), round(real_price,2), 
                            round(pred_price,2), round(price_diff,2), ref["price_movement"], 
                            action, round(profit,2), round(pred_return,2)])
        y_plot_pred[i] = pred_price
        y_plot_real[i] = real_price
    report_df = pd.DataFrame(report_data, columns=["PrevDate", "PredDate", "PrevPrice", "RealPrice", "PredPrice", "Diff", 
                                                   "PriceMove", "PredAction", "Profit", "PredReturn"]) 
    
    # Print some stats
    print("Backtesting since: {}, to: {}. Total dates: {}".format(first_eval_date, all_dates[first_eval_date_idx + total_dates_to_eval - 1], total_dates_to_eval))
    print("Profit: ${}, Buys: {}, Nothing: {}".format(
        round(report_df["Profit"].sum(), 2), 
        len(report_df[report_df["PredAction"] == "Buy"]),
        len(report_df[report_df["PredAction"] == "Nothing"])
    ))
    print("Worst loss: ${}. Total day losses: {}/{}".format(
        round(report_df["Profit"].min(), 2),
        len(report_df[report_df["Profit"] < 0]),
        len(report_df)
    ))
    print("Best profit: ${}. Total day wins: {}/{}".format(
        round(report_df["Profit"].max(), 2),
        len(report_df[report_df["Profit"] > 0]),
        len(report_df)
    ))
    
    # Plot prices
    fig, ax = plt.subplots(1,1, figsize=(20,8))
    x_plot = np.arange(total_dates_to_eval)
    plt.plot(x_plot, y_plot_real, color="green")
    plt.scatter(x_plot, y_plot_real, c="green", label="Real")
    plt.plot(x_plot, y_plot_pred, color="red")
    plt.scatter(x_plot, y_plot_pred, c="red", label="Prediction")
    plt.title("Real vs Prediction")
    plt.xlabel("Time")
    plt.ylabel("Price $")
    plt.legend()
    plt.show()
    
    # Display details
    display(HTML(report_df.to_html(index=False)))
    
# Don't use latest year for training
recent_prices_date = "2017-01-01"
old_prices, recent_prices = split_prices_by_date(df, recent_prices_date)

# Train model
WINDOW_SIZE = 260
FORWARD_DAYS = 30
EPOCHS = 100
result = test_model(old_prices, WINDOW_SIZE, FORWARD_DAYS, EPOCHS)
model = result["model"]
# Backtest
backtest(df, recent_prices_date, WINDOW_SIZE, FORWARD_DAYS, model)
show_history_size = 0.1
predict_return_by_date("sp500", df, "2017-01-03", WINDOW_SIZE, FORWARD_DAYS, model, show_history_size)
predict_return_by_date("sp500", df, "2017-02-28", WINDOW_SIZE, FORWARD_DAYS, model, show_history_size)
predict_return_by_date("sp500", df, "2017-04-03", WINDOW_SIZE, FORWARD_DAYS, model, show_history_size)
predict_return_by_date("sp500", df, "2017-06-01", WINDOW_SIZE, FORWARD_DAYS, model, show_history_size)
predict_return_by_date("sp500", df, "2017-08-01", WINDOW_SIZE, FORWARD_DAYS, model, show_history_size)
predict_return_by_date("sp500", df, "2017-10-02", WINDOW_SIZE, FORWARD_DAYS, model, show_history_size)
predict_return_by_date("sp500", df, "2017-10-10", WINDOW_SIZE, FORWARD_DAYS, model, show_history_size)
