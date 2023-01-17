!pip install apscheduler

from datetime import datetime
import time
import os

from apscheduler.schedulers.background import BackgroundScheduler

def train_model():
    print('Executando tarefa! The time is: %s' % datetime.now())
    !python Deep_Learning_Classificacao.ipynb


if __name__ == '__main__':
    scheduler = BackgroundScheduler()
    scheduler.add_job(train_model, 'interval', seconds=2)
    scheduler.start()
    print('Press Ctrl+{0} to exit'.format('Break' if os.name == 'nt' else 'C'))

    try:
        # This is here to simulate application activity (which keeps the main thread alive).
        while True:
            time.sleep(2)
    except (KeyboardInterrupt, SystemExit):
        # Not strictly necessary if daemonic mode is enabled but should be done if possible
        scheduler.shutdown()
