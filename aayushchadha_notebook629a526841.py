import sklearn

count = 0

def songVector(row):

    vector_sum = 0

    words = row.lower().split()

    for each in words:

        vector_sum = vector_sum + songs2vec[each]

    global count

    count += 1

    print ("Rows completed: " + str(count))

    return vector_sum.reshape(1,-1)







songs['vector_sum'] = songs['text'].apply(songVector)





tsne = sklearn.manifold.TSNE(n_components=2, n_iter=200, learning_rate=200, random_state=0)

def dimensionality_reduction(row):

    return tsne.fit_transform(row)

print (np.float(-0.0508137))

songs.head()

songs['xy_coords'] = songs['vector_sum'].apply(dimensionality_reduction)