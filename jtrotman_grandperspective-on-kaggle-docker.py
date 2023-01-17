import os, sys, time

import datetime

import shutil



total, used, free = shutil.disk_usage("/")

total, used, free
DATE_FORMAT = '%d/%m/%Y, %H:%M'

DATE_FORMAT_VOL = '%Y-%m-%d %H:%M:%S +0000'
scanTime = time.strftime(DATE_FORMAT_VOL)

scanTime
def dfmt(d):

    return datetime.datetime.fromtimestamp(d).strftime(DATE_FORMAT)
f = open('GrandPerspectiveScanDump.gpscan', 'w')



print(f'''<?xml version="1.0" encoding="UTF-8"?>

<GrandPerspectiveScanDump appVersion="30" formatVersion="4">

<ScanInfo volumePath="/" volumeSize="{total}" freeSpace="{free}" scanTime="{scanTime}" fileSizeMeasure="logical">''', file=f)



def walk(ipath, folder=None):

    s = os.stat(ipath)

    l = os.listdir(ipath)

    name = folder

    if name is None:

        name = os.path.abspath(ipath)

    print(f'<Folder name="{name}" created="{dfmt(s.st_ctime)}" modified="{dfmt(s.st_mtime)}" accessed="{dfmt(s.st_atime)}">', file=f)

    for e in l:

        # Files: list details

        full = os.path.join(ipath, e)

        if os.path.isfile(full):

            s = os.stat(full)

            print(f'<File name="{e}" size="{s.st_size}" created="{dfmt(s.st_ctime)}" modified="{dfmt(s.st_mtime)}" accessed="{dfmt(s.st_atime)}"/>', file=f)

    for e in l:

        # Folders: recursive call

        full = os.path.join(ipath, e)

        if os.path.isdir(full):

            walk(full, e)

    print(f"</Folder>", file=f)



walk("/opt")



print(f'''</ScanInfo>

</GrandPerspectiveScanDump>

''', file=f)



f.close()