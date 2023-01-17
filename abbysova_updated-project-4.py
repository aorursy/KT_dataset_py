petTrainShape=("pet_train data shape: " ,pet_train.shape)

petSubmitShape=("pet_submit data shape: " ,pet_submit.shape)

petSpeed=("pet_adopt_speed data shape: " ,pet_adopt_speed.shape)



AdoptionSpeed = NN.evaluate(X_test, y_test, verbose=2)



submissions=pd.DataFrame({'PetID': submit.PetID, 'Name': submit.Name, 'Age': submit.Age} )





submissions.to_csv("submission.csv", index=False, header=True)



submissions.head(10)
