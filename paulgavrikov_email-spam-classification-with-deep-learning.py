import pandas as pd

dataset = pd.read_csv('../input/emailspamdetection/SpamDetectionData.txt', sep=',', skiprows=3, header=None, names=['label', 'message'])
dataset = dataset.dropna()
dataset
dataset.label = dataset.label.map(lambda label: 1 if label == 'Spam' else 0)
dataset.message = dataset.message.map(lambda message: message.replace('<p>', '').replace('</p>', ''))
train = dataset[:2000]
test = dataset[2000:]
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

def one_hot_token(tokenizer, series):
    sequences = tokenizer.texts_to_sequences(series)
    encoded_samples = np.zeros((len(series), len(tokenizer.word_index) + 1), 'uint8')
    for i, sequence in enumerate(sequences):
        for token in sequence:
            encoded_samples[i, token] = 1
    return encoded_samples

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train.message)

y_train = train.label.astype("uint8")
y_test = test.label.astype("uint8")
X_train = pd.DataFrame(one_hot_token(tokenizer, train.message), columns=['PADDING'] + list(tokenizer.word_index.keys()))
X_test = pd.DataFrame(one_hot_token(tokenizer, test.message), columns=['PADDING'] + list(tokenizer.word_index.keys()))
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
import matplotlib.pyplot as plt

model = Sequential()
model.add(InputLayer(input_shape=(len(tokenizer.word_index) + 1, )))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, shuffle=True, epochs=10, batch_size=16, validation_split=0.2)
for key in history.history:
    plt.plot(history.history[key], label=key)
plt.legend()
plt.show()
model.evaluate(X_test, y_test)
spam_1 = """
We welcome you to MoSex Tickets.

You can now take part in the various services we have to offer you. Some of these services include:

Permanent Cart - Any products added to your online cart remain there until you remove them, or check them out.
Address Book - We can now deliver your products to another address other than yours! This is perfect to send birthday gifts direct to the birthday-person themselves.
Order History - View your history of purchases that you have made with us.
Products Reviews - Share your opinions on products with our other customers.

For help with any of our online services, please email the store-owner: store@museumofsex.com.

Note: This email address was provided on our registration page. If you own the email and did not register on our site, please send an email to store@museumofsex.com. 
"""

spam_2 = """

Hello Friend

I am very sorry in reaching out to you without a consent proper authorization, but I think it’s wise for me to share with you a business proposal of $18M about your last name for an investment transaction which is 100% risk free if interested please reply my email: nr448461@gmail.com

N C. R
"""

ham_1 = """
Dear Paul,

This confirms your submission for a refund request for one or more items through Report a Problem. A decision on the request will be updated to Report a Problem within 48 hours.

You can visit Report a Problem at any time after the 48 hour window to check the status of your submission. After logging in, click on the “Check status of claims” link near the top of the page.

Kind regards,

Apple Support 
"""

ham_2 = """
Hi Paul,

Attached, the final version.
"""

model.predict(one_hot_token(tokenizer, [spam_1, spam_2, ham_1, ham_2]))