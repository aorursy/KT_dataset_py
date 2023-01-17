import math

#A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5
# Smaller numbers
target = [30, 40, 50, 60]
twenty_percent_lesser = [24, 32, 40, 48]
twenty_percent_more = [36, 48, 60, 72]
rmsle(target, twenty_percent_lesser), rmsle(target, twenty_percent_more)
# Big numbers  *10^6
target = [30000000, 40000000, 50000000, 60000000]
twenty_percent_lesser = [24000000, 32000000, 40000000, 48000000]
twenty_percent_more = [36000000, 48000000, 60000000, 72000000]
rmsle(target, twenty_percent_lesser), rmsle(target, twenty_percent_more)
# Smaller percent - 10%
target = [30000000, 40000000, 50000000, 60000000]
ten_percent_lesser = [27000000, 36000000, 45000000, 54000000]
ten_percent_more = [33000000, 44000000, 55000000, 66000000]
rmsle(target, ten_percent_lesser), rmsle(target, ten_percent_more)
# Higher percent - 40%
target = [30000000, 40000000, 50000000, 60000000]
forty_percent_lesser = [18000000, 24000000, 30000000, 36000000]
forty_percent_more = [42000000, 56000000, 70000000, 84000000]
rmsle(target, forty_percent_lesser), rmsle(target, forty_percent_more)