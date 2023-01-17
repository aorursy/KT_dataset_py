def transform_image(X, shift):
    '''
    Return a shifted image with mirrored padding
    '''
    return np.concatenate([X, X[:, np.arange(X.shape[1]-2, 0, -1), :], X], axis=1)[:, shift:shift+X.shape[1], :]

def batch_augmentation(X, y, shift_ratio=0.3, mirror_ratio=0.35):
    '''
    Return a batch with randomized augmentation
    '''
    for i in range(X.shape[0]):
        r = np.random.random()
        if r < shift_ratio:
            shift = np.random.randint(2*X.shape[2]-1)
            X[i] = transform_image(X[i], shift)
            y[i] = transform_image(y[i], shift)
        elif r < shift_ratio+mirror_ratio:
            reversed = np.arange(X.shape[2]-1, -1, -1)
            X[i] = X[i, :, reversed, :]
            y[i] = y[i, :, reversed, :]
    return X, y

def batch_generator(X, y, batch_size=32, shift_ratio=0.3, mirror_ratio=0.35):
    '''
    Return a random batch from X, y
    '''
    left = 0
    while True:
        perm = np.random.permutation(X.shape[0])
        left = 0
        while left < perm.shape[0]:
            right = min(perm.shape[0], left+batch_size)
            idx = perm[left:right]
            yield batch_augmentation(X[idx], y[idx], shift_ratio, mirror_ratio)
            left = right
