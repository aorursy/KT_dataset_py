import pandas as pd
import numpy as np

df = pd.read_csv("../input/fics-chess-games/fics_fen_2_2M_rnn.csv", header=None, sep=";", names=["FEN"])
X_data = df["FEN"]
X_data.head()
games_sequence = list(df["FEN"][:])
len(games_sequence)
games_list = []
cut = 0
for i, pos in enumerate(games_sequence):
    if pos == "END":
        if len(games_sequence[cut:i]) > 11:
            games_list.append(games_sequence[cut:i])
        cut = i+1
games_list[34]   
gamesnr = len(games_list)
gamesnr
def batchtotensor(inputbatch):
    
    pieces_str = "PNBRQK"
    pieces_str += pieces_str.lower()
    pieces = set(pieces_str)
    valid_spaces = set(range(1,9))
    pieces_dict = {pieces_str[0]:1, pieces_str[1]:2, pieces_str[2]:3, pieces_str[3]:4 , pieces_str[4]:5, pieces_str[5]:6,
              pieces_str[6]:-1, pieces_str[7]:-2, pieces_str[8]:-3, pieces_str[9]:-4 , pieces_str[10]:-5, pieces_str[11]:-6}

    maxnum = len(inputbatch)
    boardtensor = np.zeros((maxnum, 8, 8,7))
    
    for num, inputstr in enumerate(inputbatch):
        inputliste = inputstr.split()
        rownr = 0
        colnr = 0
        for i, c in enumerate(inputliste[0]):
            if c in pieces:
                boardtensor[num, rownr, colnr, np.abs(pieces_dict[c])-1] = np.sign(pieces_dict[c])
                colnr = colnr + 1
            elif c == '/':  # new row
                rownr = rownr + 1
                colnr = 0
            elif int(c) in valid_spaces:
                colnr = colnr + int(c)
            else:
                raise ValueError("invalid fenstr at index: {} char: {}".format(i, c))
        
        if inputliste[1] == "w":
            for i in range(8):
                for j in range(8):
                    boardtensor[num, i, j, 6] = 1
        else:
            for i in range(8):
                for j in range(8):
                    boardtensor[num, i, j, 6] = -1
  
    return boardtensor
sequence_length = 1
def myGenerator():
    while 1:
        for i in range(len(games_list)): 
            X_batch = []
            y_batch = []
            for j in range(len(games_list[i])-sequence_length-1):
                X_batch.append(batchtotensor(games_list[i][j:j+sequence_length]))
                y_batch.append(batchtotensor(games_list[i][j+sequence_length:j+sequence_length+1]))
                
            X_batch = np.array(X_batch)
            y_batch = np.array(y_batch)
            y_batch = np.squeeze(y_batch, axis=1)

            yield (X_batch, y_batch)
from keras.models import Sequential
from keras.layers import Input, Dense, LSTM, Reshape, SimpleRNN
from keras.models import Model

model = Sequential()
model.add(Reshape((sequence_length,448,), input_shape=(sequence_length,8,8,7)))
model.add(SimpleRNN(400,return_sequences=True))
model.add(SimpleRNN(400))
model.add(Dense(448, activation='tanh'))
model.add(Reshape((8,8,7)))
print(model.summary())
from pickle import load
from keras.models import load_model

model = load_model('../input/model1/model_chess_1.h5')
my_generator = myGenerator()
model.compile(loss='mse', optimizer='adam')
history = model.fit_generator(my_generator, steps_per_epoch = gamesnr, epochs = 1, verbose=1,  workers=2)
model.save('model_chess_2.h5')
def tensor_to_fen(inputtensor):
    pieces_str = "PNBRQK"
    pieces_str += pieces_str.lower()
    
    maxnum = len(inputtensor)
    
    outputbatch = []
    for i in range(maxnum):
        fenstr = ""
        for rownr in range(8):
            spaces = 0
            for colnr in range(8):
                for lay in range(6):                    
                    if inputtensor[i,rownr,colnr,lay] == 1:
                        if spaces > 0:
                            fenstr += str(spaces)
                            spaces = 0
                        fenstr += pieces_str[lay]
                        break
                    elif inputtensor[i,rownr,colnr,lay] == -1:
                        if spaces > 0:
                            fenstr += str(spaces)
                            spaces = 0
                        fenstr += pieces_str[lay+6]
                        break
                    if lay == 5:
                        spaces += 1
            if spaces > 0:
                fenstr += str(spaces)
            if rownr < 7:
                fenstr += "/"
        if inputtensor[i,0,0,6] == 1:
            fenstr += " w"
        else:
            fenstr += " b"
        outputbatch.append(fenstr)
    
    return outputbatch
from pickle import load
from keras.models import load_model

model = load_model('../input/model1/model_chess_1.h5')

def generate_position(model, seed_fens):
    encoded_seed = batchtotensor(seed_fens)
    seed_list = np.array([encoded_seed])
    pos = model.predict(seed_list)
    pos = np.round(pos)
    
    return tensor_to_fen(pos)
    
prog = generate_position(model, [games_sequence[0]])
prog
count = 0
countall = 0
for i in range(len(games_sequence)-1):
    if games_sequence[i] <= 'END' and games_sequence[i+1] <= 'END':
        countall += 1
        prog = generate_position(model, [games_sequence[i]])
        if prog[0] == games_sequence[i+1]:
            count += 1
        
count / countall
seed_fens = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w",
"rnbqkbnr/pppppppp/8/8/5P2/8/PPPPP1PP/RNBQKBNR b",
"rnbqkbnr/ppp1pppp/3p4/8/5P2/8/PPPPP1PP/RNBQKBNR w",
"rnbqkbnr/ppp1pppp/3p4/8/5P2/5N2/PPPPP1PP/RNBQKB1R b",
"rnbqkb1r/ppp1pppp/3p1n2/8/5P2/5N2/PPPPP1PP/RNBQKB1R w",
"rnbqkb1r/ppp1pppp/3p1n2/8/5P2/3P1N2/PPP1P1PP/RNBQKB1R b",
"rn1qkb1r/ppp1pppp/3p1n2/8/5Pb1/3P1N2/PPP1P1PP/RNBQKB1R w",
"rn1qkb1r/ppp1pppp/3p1n2/8/4PPb1/3P1N2/PPP3PP/RNBQKB1R b",
"r2qkb1r/pppnpppp/3p1n2/8/4PPb1/3P1N2/PPP3PP/RNBQKB1R w",
"r2qkb1r/pppnpppp/3p1n2/8/4PPb1/3P1N2/PPP1B1PP/RNBQK2R b"]
seed_fens = [seed_fens[-sequence_length]]
seed_fens
fenstring = generate_position(model, seed_fens)
fenliste = fenstring[0].split()
fenliste
import chess
import chess.svg
from IPython.display import SVG

board = chess.Board(fenliste[0] + " " + fenliste[1] + " - " + " - "  "0" + " 1")
SVG(chess.svg.board(board=board,size=400))