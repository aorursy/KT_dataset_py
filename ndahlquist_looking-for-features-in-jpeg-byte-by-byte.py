import bson

import struct





def parse_jfif_header(jfif_header):

    # APP0 Marker.

    assert jfif_header[:2] == b'\xff\xe0'



    app0_len = struct.unpack('>H', jfif_header[2:4])[0]

    assert app0_len == 16



    # "JFIF" (zero terminated) Id String

    assert jfif_header[4:9] == b'JFIF\x00'



    # JFIF Format Revision

    assert jfif_header[9:11] == b'\x01\x01'



    # Units used for Resolution

    assert jfif_header[11] == 0



    horizontal_resolution = struct.unpack('>H', jfif_header[12:14])[0]

    assert horizontal_resolution == 1



    vertical_resolution = struct.unpack('>H', jfif_header[14:16])[0]

    assert vertical_resolution == 1



    x_thumbnail = jfif_header[16]

    assert x_thumbnail == 0

    y_thumbnail = jfif_header[17]

    assert y_thumbnail == 0



    return jfif_header[app0_len+2:]





def parse_quantization_table(quantization_table, expected_table_num):

    # SOI / Start of image marker.

    assert quantization_table[:2] == b'\xff\xdb'



    quantization_table_length = struct.unpack('>H', quantization_table[2:4])[0]

    assert quantization_table_length == 67



    # Quantization table information (1 table, 8-bit precision).

    info = quantization_table[4]

    table_num = info & 0b00000111

    assert(table_num == expected_table_num)

    precision = info & 0b11111000

    assert(precision == 0)



    return quantization_table[quantization_table_length+2:]





def parse_frame(frame):

    # SOF0 Marker (Start Of Frame)

    assert frame[:2] == b'\xff\xc0'



    sof0_len = struct.unpack('>H', frame[2:4])[0]

    assert sof0_len == 17



    # Data precision (8 bits)

    assert frame[4] == 8



    image_height = struct.unpack('>H', frame[5:7])[0]

    assert image_height == 180

    image_width = struct.unpack('>H', frame[7:9])[0]

    assert image_width == 180



    number_of_components = frame[10]

    assert number_of_components == 1  # Grey scale



    component_id = frame[11]

    assert component_id == 34

    sampling_factors = frame[12]

    assert sampling_factors == 0

    quant_table_num = frame[13]

    assert quant_table_num == 2



    return frame[sof0_len+2:]





def parse_huffman_table(huffman_table, expected_table_num, expected_table_type):

    """

    See http://www.impulseadventure.com/photo/jpeg-huffman-coding.html for details.

    """



    # DHT( Define Huffman Table) marker

    assert huffman_table[:2] == b'\xff\xc4'



    dht_len = struct.unpack('>H', huffman_table[2:4])[0]

    if expected_table_type == 0:

        assert dht_len == 31

    else:

        assert dht_len == 181



    info = huffman_table[4]

    table_num = info & 0b00000111

    assert table_num == expected_table_num

    table_type = (info & 0b00010000) >> 4

    assert table_type == expected_table_type

    # Bits 5-7 must be 0.

    assert info & 0b11100000 == 0



    return huffman_table[dht_len + 2:]





def parse_scan(scan):



    # SOS (Start Of Scan) marker

    assert scan[:2] == b'\xff\xda'



    scan_len = struct.unpack('>H', scan[2:4])[0]

    assert scan_len == 12



    num_components = scan[4]

    assert num_components == 3



    for i in range(num_components):

        component_id = scan[5 + 2 * i]

        assert component_id == i + 1

        huffman_table = scan[5 + 2 * i + 1]

        if i == 0:

            assert huffman_table == 0

        else:

            assert huffman_table == 17





def inspect_jpeg_file_layout(bytestream):



    # SOI (Start Of Image) marker.

    assert bytestream[:2] == b'\xff\xd8'

    remaining = bytestream[2:]



    remaining = parse_jfif_header(remaining)

    remaining = parse_quantization_table(remaining, 0)

    remaining = parse_quantization_table(remaining, 1)

    remaining = parse_frame(remaining)



    # Used for DC component of Luminance

    remaining = parse_huffman_table(remaining, 0, 0)



    # Used for AC component of Luminance

    remaining = parse_huffman_table(remaining, 0, 1)



    # Used for DC component of Chrominance

    remaining = parse_huffman_table(remaining, 1, 0)



    # Used for AC component of Chrominance

    remaining = parse_huffman_table(remaining, 1, 1)

    parse_scan(remaining)



    # End of scan

    assert bytestream[-2:] == b'\xff\xd9'

from tqdm import tqdm_notebook



with open('../input/train_example.bson', 'rb') as f:

    data = bson.decode_file_iter(f)



    for d in data:

        for img in d['imgs']:

            inspect_jpeg_file_layout(img['picture'])

    

    print("All files are clean!")
with open('../input/test.bson', 'rb') as f:

    data = bson.decode_file_iter(f)

    

    bar = tqdm_notebook(total=1768182)

    for d in data:

        bar.update()

        

        for img in d['imgs']:

            inspect_jpeg_file_layout(img['picture'])

    

    print("All files are clean!")
with open('../input/train.bson', 'rb') as f:

    data = bson.decode_file_iter(f)

    

    bar = tqdm_notebook(total=7069896)

    for d in data:

        bar.update()

        

        for img in d['imgs']:

            inspect_jpeg_file_layout(img['picture'])

    

    print("All files are clean!")