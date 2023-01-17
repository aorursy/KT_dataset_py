import pandas
import numpy as np

dt = { 'acoustic_data': 'i2', 'time_to_failure': 'f8' }
data = pandas.read_csv("../input/train.csv", dtype=dt, engine='c', low_memory=True)

N = data.shape[0]
print("Data size", N)
#print("Data stats", data.describe())
digits = 10

oldValue = data.iloc[0,1]
newValue = data.iloc[1,1]
frame_diff = round(oldValue - newValue, digits)
oldDiff = frame_diff
print("Time to failure", round(oldValue, digits))
for i in range(1, 20000):
    newValue = data.iloc[i,1]
    newDiff = round(oldValue - newValue, digits)
    if oldDiff != newDiff:
        print("Time difference changed from", oldDiff, "to", newDiff, "on frame", i)
        oldDiff = newDiff
    oldValue = newValue
print("Time to failure", round(newValue, digits))
chunk_diffs = [
    round(data.iloc[0,1] - data.iloc[4096,1], digits),
    round(data.iloc[4095,1] - data.iloc[8191,1], digits)
]
print(chunk_diffs)
M = 4096
C = N//M+1
R = C*M - N
print("Chunk size", M)
print("Number of chunks", C)
print("Number of incomplete chunks", R)
chunk_digits = 8

max_error = 1e-9
small_error_count = 10
failure_starts = [[0,0]]
failure_index = 0
chunks = np.zeros((C, 2))
chunk_index = 0
startValue = data.iloc[0,1]
chunks[chunk_index] = [0, round(startValue, digits)]
print("Time to", failure_index, "earthquake", round(startValue, digits))
for i in range(1, N):
    j = i - chunks[chunk_index,0]
    newValue = data.iloc[i,1]
    newDiff = round(startValue - newValue, digits)
    frame_error = newDiff - round(frame_diff * j, digits)
    if newDiff < 0:
        print("New earthquake after", chunk_index+1, "chunks", i-failure_starts[failure_index][0], "samples")
        data.iloc[failure_starts[failure_index][0]:i,0].to_csv('failure{0}_data.zip'.format(failure_index), index=False, compression="zip")
        np.savetxt('failure{0}_chunks.csv'.format(failure_index), chunks[failure_starts[failure_index][1]:chunk_index+1], fmt='%d, %f', header='Frame, Time', comments='')
        print("Time to", failure_index, "earthquake", round(startValue, digits))
        failure_index += 1
        chunk_index += 1
        failure_starts.append([i, chunk_index])
        startValue = newValue
        chunks[chunk_index] = [i, round(startValue, digits)]
        print("Time to", failure_index, "earthquake", round(startValue, digits))
    elif round(newDiff, chunk_digits) in chunk_diffs:
        # 
        chunk_index += 1
        chunks[chunk_index] = [i, newDiff]
        startValue = newValue
        if chunk_index % 1000 == 0:
            print("Chunk", chunk_index)
    elif frame_error != 0:
        if small_error_count > 0:
            print("Unexpected frame", failure_index, chunk_index, j, "duration", newDiff, "expected", round(frame_diff * j, digits))
            small_error_count -= 1
        if frame_error > max_error:
            print("Prediction error, stopping execution")
            break

data.iloc[failure_starts[failure_index][0]:,0].to_csv('failure{0}_data.zip'.format(failure_index), index=False, compression="zip")
np.savetxt('failure{0}_chunks.csv'.format(failure_index), chunks[failure_starts[failure_index][1]:], fmt='%d, %f', header='Frame, Time', comments='')
print("Time to", failure_index, "earthquake", round(newValue, digits))
failure_count = failure_index + 1
chunk_count = chunk_index + 1
print("Chunk count expected", C, "actual", chunk_count)
for fi in range(failure_count):
    first_frame = failure_starts[fi][0]
    last_frame = int(chunks[(failure_starts[fi+1][1] if fi < failure_count - 1 else chunk_count) - 1, 0])
    first_ttf = data.iloc[first_frame,1]
    last_ttf = data.iloc[last_frame,1]
    rate = (last_frame - first_frame)/(first_ttf - last_ttf)
    print("Earthquake", failure_index, "frame range", first_frame, last_frame, "time range", first_ttf, last_ttf, "measured frame rate", rate, "Hz")