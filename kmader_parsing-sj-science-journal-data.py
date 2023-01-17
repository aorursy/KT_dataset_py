!git clone https://github.com/kmader/parse_sj
import os, sys
sys.path.append(os.path.join('parse_sj', 'fmt'))
import sensor_pb2, experiment_pb2
!unzip -q ../input/*.sj
!ls -lh *.proto
with open('experiment.proto', 'rb') as f:
    experiment_info = experiment_pb2.Experiment()
    experiment_info.ParseFromString(f.read())
experiment_info
with open('sensorData.proto', 'rb') as f:
    read_sensor = sensor_pb2.SensorData()
    read_sensor.ParseFromString(f.read())
read_sensor
read_sensor.ListFields()
!rm *.proto
!rm -rf parse_sj