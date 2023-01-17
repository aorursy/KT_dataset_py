# install the kaggle binary and add your credentials to the environment

!pip install kaggle --quiet

%env KAGGLE_USERNAME=jorijnsmit

%env KAGGLE_KEY=aLlYoUrBaSeArEbElOnGtOuS
# upload some .csv file

!kaggle competitions submit your-competition-name -f submission.csv -m 'My submission message'
# list your submissions

!kaggle competitions submissions your-competition-name
# show the top leaderboard

!kaggle competitions leaderboard --show your-competition-name
# etc