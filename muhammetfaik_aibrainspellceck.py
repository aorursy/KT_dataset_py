!pip install txt2txt

!wget aibrian.com/checkpoint

!wget aibrian.com/params
from txt2txt import build_model, infer

import time

model, params = build_model(params_path='params', enc_lstm_units=256)

model.load_weights('checkpoint')
start = time.time()

print(infer("i will be there for you", model, params))

end = time.time()

print(end - start)



start = time.time()

print(infer(["i will be there for you","these is not that gre at","i dotlike this","i work at reckonsys"], model, params))

end = time.time()

print(end - start)



start = time.time()

print(infer("i will be there for you", model, params))

end = time.time()

print(end - start)