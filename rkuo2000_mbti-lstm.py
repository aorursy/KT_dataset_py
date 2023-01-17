#!git clone https://github.com/ianscottknight/Predicting-Myers-Briggs-Type-Indicator-with-Recurrent-Neural-Networks

!git clone https://github.com/rkuo2000/Predicting-Myers-Briggs-Type-Indicator-with-Recurrent-Neural-Networks

%cd Predicting-Myers-Briggs-Type-Indicator-with-Recurrent-Neural-Networks
!python separate_clean_and_unclean.py
!python make_training_set.py
!python make_test_set.py
!python rnn.py
!python average_prob_predictor.py
!cp "trump prediction"/trumptweets.csv .
!python "trump prediction"/predictor.py