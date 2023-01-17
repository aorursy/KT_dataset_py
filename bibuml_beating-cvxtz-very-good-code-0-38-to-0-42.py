from sklearn.metrics.pairwise import euclidean_distances

test_files = glob.glob("Whales/test/*.jpg")
l_image_name_test = [test_files[i].split('\\')[1] for i in range(len(test_files))]
l_class_data = [data['Id'][i] for i in range(len(data))]               # data = file "train.csv"

# test_preds = predict of inference model for test images (data = for "train.csv" images)
test_image_dist_all = euclidean_distances(test_preds, data_preds)      
preds_str = []


for ind in range(len(l_image_name_test)) :
    test_image_dist = test_image_dist_all[ind]     # distances between the test image and all the 'train.csv' images
    vect_dist = [(l_class_data[i],test_image_dist[i]) for i in range(len(test_image_dist))]    # create list of couples (class, distance)
    vect_dist.append(("new_whale", 0.0))  # add "new_whale" ecach time
    vect_dist.sort(key=lambda x: x[1])    # sort  in order to have first the nearest
    vect_dist = vect_dist[0:50]            # best 50 nearest 
    
    vect_classes = [vect_dist[i][0] for i in range(len(vect_dist))]
    # Maintain only one occurrence per class
    vect_result = [vect_dist[0]] + [vect_dist[i] for i in range(1,len(vect_dist)) if vect_classes[i] not in vect_classes[0:i]]
    vect_result = vect_result[:5]   # take fist 5 nearest
    preds_str.append(" ".join([x[0] for x in vect_result]))